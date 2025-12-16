import os
import cv2
import imageio
from gaussian_renderer import render_fn_dict
import numpy as np
import torch
from pbr import CubemapLight, get_brdf_lut
from scene import GaussianModel, Scene
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.cameras import Camera
from scene.transfer_mlp import TransferMLP
from utils.graphics_utils import focal2fov, fov2focal 
from torchvision.utils import save_image
import torchvision
from tqdm import tqdm
from lpipsPyTorch import get_lpips_model
from utils.loss_utils import ssim
from utils.image_utils import psnr
from scene.utils import load_img_rgb
import warnings
from utils.graphics_utils import read_hdr, latlong_to_cubemap, latlong_to_cubemap_orb, latlong_to_cubemap_nvdiffrec
from utils.general_utils import load_json_config

warnings.filterwarnings("ignore")



def env_map_to_cam_to_world_by_convention(envmap: np.ndarray, c2w, convention):
    R = c2w[:3,:3]
    H, W = envmap.shape[:2]
    theta, phi = np.meshgrid(np.linspace(-0.5*np.pi, 1.5*np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([-np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)],
                           axis=-1).reshape(H*W, 3)    # [H, W, 3]
    viewdirs = (R.T @ viewdirs.T).T.reshape(H, W, 3)
    viewdirs = viewdirs.reshape(H, W, 3)
    # This is correspond to the convention of +Z at left, +Y at top
    # -np.cos(theta) * np.sin(phi), np.cos(phi), -np.sin(theta) * np.sin(phi)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped = cv2.remap(envmap, coord_x, coord_y, cv2.INTER_LINEAR)
    if convention == 'ours':
        return envmap_remapped
    if convention == 'physg':
        # change convention from ours (Left +Z, Up +Y) to physg (Left -Z, Up +Y)
        envmap_remapped_physg = np.roll(envmap_remapped, W//2, axis=1)
        return envmap_remapped_physg
    if convention == 'nerd':
        # change convention from ours (Left +Z-X, Up +Y) to nerd (Left +Z+X, Up +Y)
        envmap_remapped_nerd = envmap_remapped[:,::-1,:]
        return envmap_remapped_nerd

    assert convention == 'invrender', convention
    # change convention from ours (Left +Z-X, Up +Y) to invrender (Left -X+Y, Up +Z)
    theta, phi = np.meshgrid(np.linspace(1.0 * np.pi, -1.0 * np.pi, W), np.linspace(0., np.pi, H))
    viewdirs = np.stack([np.cos(theta) * np.sin(phi),
                         np.sin(theta) * np.sin(phi),
                         np.cos(phi)], axis=-1)    # [H, W, 3]
    # viewdirs = np.stack([-viewdirs[...,0], viewdirs[...,2], viewdirs[...,1]], axis=-1)
    coord_y = ((np.arccos(viewdirs[..., 1])/np.pi*(H-1)+H)%H).astype(np.float32)
    coord_x = (((np.arctan2(viewdirs[...,0], -viewdirs[...,2])+np.pi)/2/np.pi*(W-1)+W)%W).astype(np.float32)
    envmap_remapped_Inv = cv2.remap(envmap_remapped, coord_x, coord_y, cv2.INTER_LINEAR)
    return envmap_remapped_Inv

if __name__ == '__main__':
    # Set up command line argument parser
    parser = ArgumentParser(description="Composition and Relighting for Relightable 3D Gaussian")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument('-e', '--envmap_path', default="/data/zhouyongyang/dataset/tensorIR/env_maps/high_res_envmaps_1k/", help="Env map path")
    parser.add_argument('-bg', "--background_color", type=float, default=1,
                        help="If set, use it as background color")
    parser.add_argument('-t', '--type', choices=['render_ref', 'render_ref_pbr', 'render_ref_fast'], default='render_ref')
    parser.add_argument("--occlusion_path", type=str, default=None)
    parser.add_argument("--no_rescale_albedo", action="store_true")
    parser.add_argument("--skip_save_image", action="store_true", default=False)
    parser.add_argument("--save_name", type=str, default="test_rli")
    parser.add_argument("--save_video", action='store_true', default=False)

    args = get_combined_args(parser)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    LPIPS = get_lpips_model(net_type='vgg').cuda()

    save_image_flag = not args.skip_save_image

    # load gaussians
    gaussians = GaussianModel(model.sh_degree, render_type=args.type)
    
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        iteration = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=False)

        transfer_net = None
        if pipe.compute_with_prt:
            transfer_net = TransferMLP(sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres)
            # transfer_net.training_setup(opt)
            transfer_net_checkpoint = os.path.dirname(args.checkpoint) + "/transfer_net_" + os.path.basename(args.checkpoint)
            if os.path.exists(transfer_net_checkpoint):
                transfer_net.create_from_ckpt(transfer_net_checkpoint)
                print("Successfully loaded transfer net!")
            else:
                print("Failed to load transfer net!")
        
        occlusion_volumes = torch.load(args.occlusion_path)
        bound = occlusion_volumes["bound"]
        scene = Scene(dataset, gaussians)
        canonical_rays = scene.get_canonical_rays()
        aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
        brdf_lut = get_brdf_lut().cuda()
    else:
        raise NotImplementedError
    

    if not pipe.specular_workflow:
        capture_list = ["pbr", "base_color", "diffuse_pbr", "specular_pbr"]
    else:
        capture_list = ["pbr", "diffuse_color", "specular_color", "diffuse_pbr", "specular_pbr"]

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict[args.type]

        

    test_transforms_file = os.path.join(args.source_path, "transforms_novel.json")
    contents = load_json_config(test_transforms_file)
    # fovx = contents["camera_angle_x"]
    frames = contents["frames"]
        
    psnr_pbr = 0.0
    ssim_pbr = 0.0
    lpips_pbr = 0.0
    
    psnr_albedo = 0.0
    ssim_albedo = 0.0
    lpips_albedo = 0.0
    
    mse_roughness = 0.0
    video_dict = {}
    video_dict["gt"] = []
    results_dir = os.path.join(args.model_path, args.save_name)
    os.makedirs(results_dir, exist_ok=True)

    # if not args.no_rescale_albedo:
    #     gt_albedo_list = []
    #     reconstructed_albedo_list = []
    #     albedo_gt_exist = True
    #     gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    #     for idx, frame in enumerate(tqdm(frames, leave=False)):
    #         # NeRF 'transform_matrix' is a camera-to-world transform
    #         c2w = np.array(frame["transform_matrix"])
    #         # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
    #         c2w[:3, 1:3] *= -1

    #         # get the world-to-camera transform and set R, T
    #         w2c = np.linalg.inv(c2w)
    #         R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    #         T = w2c[:3, 3]

    #         albedo_path = os.path.join(args.source_path, frame["file_path"].split("rgba")[0] + "albedo.png")
    #         if os.path.exists(albedo_path):
    #             albedo_rgba = load_img_rgb(albedo_path)
    #             fovx = frame["camera_angle_x"]
    #             # albedo_rgba[..., 0:3] = srgb_to_rgb(albedo_rgba[..., 0:3])
    #             mask = albedo_rgba[..., 3] > 0
    #             gt_albedo = torch.from_numpy(albedo_rgba[..., :3]).float().cuda()
    #             gt_albedo_list.append(gt_albedo[mask])
    #             H = albedo_rgba.shape[0]
    #             W = albedo_rgba.shape[1]
    #             fovy = focal2fov(fov2focal(fovx, W), H)
    #             custom_cam = Camera(colmap_id=0, R=R, T=T,
    #                                 FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
    #                                 image=torch.zeros(3, H, W), image_name=None, uid=0)
                

    #             cubemap = None
    #             envmap_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "envmap") + '.exr')
    #             hdri = read_hdr(envmap_path)
    #             hdri = torch.from_numpy(hdri).cuda()
    #             res = 256
    #             cubemap = CubemapLight(base_res=res).cuda()
    #             cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
    #             cubemap.build_mips()
    #             cubemap.eval()
    #             env_image = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    #             envmap_path = os.path.join(args.model_path, 'test_rli', 'envmap.png')

    #             env_image_diffuse = cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    #             envmap_path = os.path.join(args.model_path, 'test_rli', 'envmap_diffuse.png')
    #             torchvision.utils.save_image(env_image_diffuse, envmap_path)


    #             render_kwargs = {
    #                 "pc": gaussians,
    #                 "pipe": pipe,
    #                 "bg_color": background,
    #                 "is_training": False,
    #                 "dict_params": {
    #                     "transfer_net": transfer_net,
    #                     "occlusion_volumes": occlusion_volumes,
    #                     "aabb": aabb,
    #                     "cubemap": cubemap,
    #                     "refmap": cubemap,
    #                     "brdf_lut": brdf_lut,
    #                     "canonical_rays": canonical_rays,
    #                     "iteration": iteration,
    #                     "relight": True
    #                 },
    #             }


    #             with torch.no_grad():
    #                 render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)
    #             if not pipe.specular_workflow:
    #                 render_albedo = render_pkg['base_color'].permute(1, 2, 0).clamp(1e-6, 1)
    #             else:
    #                 render_albedo = (render_pkg['diffuse_color'] + render_pkg['specular_color']).permute(1, 2, 0).clamp(1e-6, 1)
    #             reconstructed_albedo_list.append(render_albedo[mask])
    #         else:
    #             albedo_gt_exist = False
    #             video_dict["gt_albedo"] = None

                    
    #     if albedo_gt_exist:
    #         gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
    #         albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
    #         albedo_scale, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)  # [3]
    #         gaussians.base_color_scale = albedo_scale
    #         print("Albedo scale:", albedo_scale)
    #     else:
    #         gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    #         print("Albedo scale:", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"))
    # else:
    #     gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    #     print("Albedo scale:", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"))

        
    for idx, frame in enumerate(tqdm(frames, leave=False)):
        base_dir = os.path.dirname(args.source_path)
        fovx = frame["camera_angle_x"]
        scene_name = frame["scene_name"]

        image_path = os.path.join(args.source_path, frame["file_path"]  + ".png")
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = np.array(frame["transform_matrix"])
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        image_rgba = load_img_rgb(image_path)
        image = image_rgba[..., :3]
        mask = image_rgba[..., 3:]
        gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
        H = image.shape[0]
        W = image.shape[1]
        fovy = focal2fov(fov2focal(fovx, W), H)
        custom_cam = Camera(colmap_id=0, R=R, T=T,
                            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                            image=torch.zeros(3, H, W), image_name=None, uid=0)
            
        gt_image = gt_image * mask + bg * (1 - mask)
        # load env map
        envmap_path = os.path.join(args.source_path, frame["file_path"].replace("rgba", "envmap") + '.exr')
        cubemap = None
        hdri = read_hdr(envmap_path)
        

        #! rotate light
        # def rotate_x(a, device=None):
        #     s, c = np.sin(a), np.cos(a)
        #     return torch.tensor([[1,  0, 0, 0], 
        #                  [0,  c, s, 0], 
        #                  [0, -s, c, 0], 
        #                  [0,  0, 0, 1]], dtype=torch.float32, device=device)

        # c2w = np.array(frame["transform_matrix"])
        # w2c = torch.linalg.inv(torch.tensor(c2w, dtype=torch.float32))
        # w2c = w2c @ rotate_x(-np.pi / 2)
        # c2w = torch.linalg.inv(w2c).numpy()
        # hdri = env_map_to_cam_to_world_by_convention(hdri, c2w, "ours")
        #! rotate light

        
        hdri = torch.from_numpy(hdri).cuda()

        res = 256
        cubemap = CubemapLight(base_res=res).cuda()
        # cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
        cubemap.base.data = latlong_to_cubemap_orb(hdri, [res, res])
        # cubemap.base.data = latlong_to_cubemap_nvdiffrec(hdri, [res, res])
        cubemap.build_mips()
        cubemap.eval()
        env_image = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, 'test_rli', 'envmap.png')
        torchvision.utils.save_image(env_image, envmap_path)
        env_image_diffuse = cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, 'test_rli', 'envmap_diffuse.png')
        torchvision.utils.save_image(env_image_diffuse, envmap_path)

        env_image_full = cubemap.export_envmap(return_img=True, base=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        torchvision.utils.save_image(env_image_full, "/data/zhouyongyang/project/RTR-GS/test_envmap/test.png")
        render_kwargs = {
            "pc": gaussians,
            "pipe": pipe,
            "bg_color": background,
            "is_training": False,
            "dict_params": {
                "transfer_net": transfer_net,
                "occlusion_volumes": occlusion_volumes,
                "aabb": aabb,
                "cubemap": cubemap,
                "refmap": cubemap,
                "brdf_lut": brdf_lut,
                "canonical_rays": canonical_rays,
                "iteration": iteration,
                "relight": True
            },
        }
        with torch.no_grad():
            render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)

        
        task_dir = os.path.join(results_dir, scene_name)

        os.makedirs(task_dir, exist_ok=True)
        os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
        for capture_type in capture_list:
            capture_type_dir = os.path.join(task_dir, capture_type)
            os.makedirs(capture_type_dir, exist_ok=True)
            video_dict[capture_type] = []

        if save_image_flag:
            save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
            video_gt_image = torch.clamp(gt_image, 0.0, 1.0).permute(1,2,0).detach().cpu()
            video_gt_image = (video_gt_image.numpy() * 255).astype('uint8')
            video_dict['gt'].append(video_gt_image)

            for capture_type in capture_list:
                if capture_type == "normal":
                    render_pkg[capture_type] = render_pkg[capture_type] * 0.5 + 0.5
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["roughness", "diffuse", "specular"]:
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["base_color"]:
                    render_pkg[capture_type] = render_pkg[capture_type] * mask + (1 - mask) * bg
                elif capture_type in ["pbr"]:
                    render_pkg[capture_type] = render_pkg["pbr"] * mask + (1 - mask) * bg
                elif capture_type in ["pbr_env"]:
                    render_pkg[capture_type] = render_pkg["pbr"] * mask + (1 - mask) * render_pkg["env_only"]
                elif capture_type in ["diffuse_pbr", "specular_pbr", "metallic", "roughness"]:
                    render_pkg[capture_type] = render_pkg["vis_dict"][capture_type] * mask + (1 - mask) * bg
                save_image(render_pkg[capture_type], os.path.join(task_dir, capture_type, f"{idx}.png"))
                video_image = torch.clamp(render_pkg[capture_type], 0.0, 1.0).permute(1,2,0).detach().cpu()
                video_image = (video_image.numpy() * 255).astype('uint8')
                video_dict[capture_type].append(video_image)

        with torch.no_grad():
            psnr_pbr += psnr(render_pkg['pbr'], gt_image).mean().double()
            ssim_pbr += ssim(render_pkg['pbr'], gt_image).mean().double()
            lpips_pbr += LPIPS(render_pkg['pbr'], gt_image).mean().double()
        
    psnr_pbr /= len(frames)
    ssim_pbr /= len(frames)
    lpips_pbr /= len(frames)
    psnr_albedo /= len(frames)
    ssim_albedo /= len(frames)
    lpips_albedo /= len(frames)
    with open(os.path.join(results_dir, f"metric.txt"), "w") as f:
        f.write(f"psnr_pbr: {psnr_pbr}\n")
        f.write(f"ssim_pbr: {ssim_pbr}\n")
        f.write(f"lpips_pbr: {lpips_pbr}\n")
        f.write(f"psnr_albedo: {psnr_albedo}\n")
        f.write(f"ssim_albedo: {ssim_albedo}\n")
        f.write(f"lpips_albedo: {lpips_albedo}\n")
    
    print("\nEvaluating: PSNR_PBR {} SSIM_PBR {} LPIPS_PBR {}".format(psnr_pbr, ssim_pbr, lpips_pbr))
    
    if args.save_video:
        video_path = os.path.join(task_dir, "video")
        os.makedirs(video_path, exist_ok=True)
        for key in video_dict:
            if video_dict[key] != None:
                imageio.mimsave(os.path.join(video_path, f"{key}_video.mp4"), np.stack(video_dict[key]), fps=24, macro_block_size=1)