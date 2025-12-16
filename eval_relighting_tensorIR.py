import os

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
from utils.graphics_utils import read_hdr, latlong_to_cubemap
from utils.general_utils import load_json_config


warnings.filterwarnings("ignore")



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
        
    # deal with each item
    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)

    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    capture_list = ["pbr", "base_color", "diffuse_pbr", "specular_pbr"]

    envmap_base_dir = args.envmap_path
    task_dict = {
        "sunset": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "sunset.hdr",
        },
        "bridge": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "bridge.hdr",
        },
        "city": {
            "capture_list": capture_list,
            "envmap_path":  envmap_base_dir + "city.hdr",
        },
        "fireplace": {
            "capture_list": capture_list,
            "envmap_path":  envmap_base_dir + "fireplace.hdr",
        },
        "forest": {
            "capture_list": capture_list,
            "envmap_path": envmap_base_dir + "forest.hdr",
        },
        "night": {
            "capture_list": capture_list,
            "envmap_path":  envmap_base_dir + "night.hdr",
        }
    }

    bg = 1 if dataset.white_background else 0
    background = torch.tensor([bg, bg, bg], dtype=torch.float32, device="cuda")
    render_fn = render_fn_dict[args.type]
    

    results_dir = os.path.join(args.model_path, args.save_name)
    task_names = ['bridge', 'city', 'fireplace', 'forest', 'night']
    for task_name in task_names:
        task_dir = os.path.join(results_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)
        # light = EnvLight(path=task_dict[task_name]["envmap_path"], scale=1)

        cubemap = None
        hdri = read_hdr(task_dict[task_name]["envmap_path"])
        hdri = torch.from_numpy(hdri).cuda()
        res = 256
        cubemap = CubemapLight(base_res=res).cuda()
        cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
        cubemap.build_mips()
        cubemap.eval()
        env_image = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, 'test_rli', 'envmap.png')

        env_image_diffuse = cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, 'test_rli', 'envmap_diffuse.png')
        torchvision.utils.save_image(env_image_diffuse, envmap_path)


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


        psnr_pbr = 0.0
        ssim_pbr = 0.0
        lpips_pbr = 0.0
        
        psnr_albedo = 0.0
        ssim_albedo = 0.0
        lpips_albedo = 0.0
        
        mse_roughness = 0.0

        video_dict = {}
        video_dict["gt"] = []
        video_dict["gt_albedo"] = []
        
        capture_list = task_dict[task_name]["capture_list"]
        for capture_type in capture_list:
            capture_type_dir = os.path.join(task_dir, capture_type)
            os.makedirs(capture_type_dir, exist_ok=True)
            video_dict[capture_type] = []


        os.makedirs(os.path.join(task_dir, "gt"), exist_ok=True)
        # os.makedirs(os.path.join(task_dir, "gt_albedo"), exist_ok=True)
        # os.makedirs(os.path.join(task_dir, "gt_pbr_env"), exist_ok=True)
        envname = os.path.splitext(os.path.basename(task_dict[task_name]["envmap_path"]))[0]

        if not args.no_rescale_albedo:
            gt_albedo_list = []
            reconstructed_albedo_list = []
            albedo_gt_exist = True
            gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            for idx, frame in enumerate(tqdm(frames, leave=False)):
                image_path = os.path.join(args.source_path, frame["file_path"] + "_" + envname + ".png")
                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(frame["transform_matrix"])
                # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
                c2w[:3, 1:3] *= -1

                # get the world-to-camera transform and set R, T
                w2c = np.linalg.inv(c2w)
                R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
                T = w2c[:3, 3]

                albedo_path = os.path.join(args.source_path, frame["file_path"].split("rgba")[0] + "albedo.png")
                if os.path.exists(albedo_path):
                    albedo_rgba = load_img_rgb(albedo_path)
                    # albedo_rgba[..., 0:3] = srgb_to_rgb(albedo_rgba[..., 0:3])
                    mask = albedo_rgba[..., 3] > 0
                    gt_albedo = torch.from_numpy(albedo_rgba[..., :3]).float().cuda()

                    gt_albedo_list.append(gt_albedo[mask])


                    H = albedo_rgba.shape[0]
                    W = albedo_rgba.shape[1]
                    fovy = focal2fov(fov2focal(fovx, W), H)

                    custom_cam = Camera(colmap_id=0, R=R, T=T,
                                        FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                                        image=torch.zeros(3, H, W), image_name=None, uid=0)

                    with torch.no_grad():
                        render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)

                    render_albedo = render_pkg['base_color'].permute(1, 2, 0).clamp(1e-6, 1)
                    
                    reconstructed_albedo_list.append(render_albedo[mask])
                else:
                    albedo_gt_exist = False
                    video_dict["gt_albedo"] = None

                    
            if albedo_gt_exist:
                gt_albedo_all = torch.cat(gt_albedo_list, dim=0)
                albedo_map_all = torch.cat(reconstructed_albedo_list, dim=0)
                albedo_scale, _ = (gt_albedo_all / albedo_map_all.clamp(min=1e-6)).median(dim=0)  # [3]
                gaussians.base_color_scale = albedo_scale
                print("Albedo scale:", albedo_scale)
            else:
                gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
                print("Albedo scale:", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"))
        else:
            gaussians.base_color_scale = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
            print("Albedo scale:", torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda"))

        
        
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(args.source_path, frame["file_path"] + "_" + envname + ".png")
            if not os.path.exists(image_path):
                image_path = os.path.join(args.source_path, frame["file_path"] + ".png")

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

            albedo_path = os.path.join(args.source_path, frame["file_path"].split("rgba")[0] + "albedo.png")
            if os.path.exists(albedo_path):
                albedo_rgba = load_img_rgb(albedo_path)
                gt_albedo = torch.from_numpy(albedo_rgba[..., :3]).permute(2, 0, 1).float().cuda()
                gt_albedo = gt_albedo * mask + bg * (1 - mask)
                # save_image(gt_albedo, os.path.join(task_dir, "gt_albedo", f"{idx}.png"))

            # gt_image_env = gt_image * mask + render_pkg["env_only"] * (1 - mask)
            # save_image(gt_image_env, os.path.join(task_dir, "gt_pbr_env", f"{idx}.png"))

            with torch.no_grad():
                render_pkg = render_fn(viewpoint_camera=custom_cam, **render_kwargs)

            if save_image_flag:
                save_image(gt_image, os.path.join(task_dir, "gt", f"{idx}.png"))
                
                video_gt_image = torch.clamp(gt_image, 0.0, 1.0).permute(1,2,0).detach().cpu()
                video_gt_image = (video_gt_image.numpy() * 255).astype('uint8')
                video_dict['gt'].append(video_gt_image)

                if os.path.exists(albedo_path):
                    video_gt_albedo = torch.clamp(gt_albedo, 0.0, 1.0).permute(1,2,0).detach().cpu()
                    video_gt_albedo = (video_gt_albedo.numpy() * 255).astype('uint8')
                    video_dict['gt_albedo'].append(video_gt_albedo)

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

                if os.path.exists(albedo_path):
                    psnr_albedo += psnr(render_pkg['base_color'], gt_albedo).mean().double()
                    ssim_albedo += ssim(render_pkg['base_color'], gt_albedo).mean().double()
                    lpips_albedo += LPIPS(render_pkg['base_color'], gt_albedo).mean().double()

                # mse_roughness += ((render_pkg['roughness'] - gt_roughness)**2).mean().double()
            


        psnr_pbr /= len(frames)
        ssim_pbr /= len(frames)
        lpips_pbr /= len(frames)
        
        psnr_albedo /= len(frames)
        ssim_albedo /= len(frames)
        lpips_albedo /= len(frames)
        

        with open(os.path.join(results_dir, f"metric_{task_name}.txt"), "w") as f:
            f.write(f"psnr_pbr: {psnr_pbr}\n")
            f.write(f"ssim_pbr: {ssim_pbr}\n")
            f.write(f"lpips_pbr: {lpips_pbr}\n")
            f.write(f"psnr_albedo: {psnr_albedo}\n")
            f.write(f"ssim_albedo: {ssim_albedo}\n")
            f.write(f"lpips_albedo: {lpips_albedo}\n")

        print("\nEvaluating {}: PSNR_PBR {} SSIM_PBR {} LPIPS_PBR {}".format(task_name, psnr_pbr, ssim_pbr, lpips_pbr))
        print("\nEvaluating {}: PSNR_ALBEDO {} SSIM_ALBEDO {} LPIPS_ALBEDO {}".format(task_name, psnr_albedo, ssim_albedo, lpips_albedo))



        if args.save_video:
            video_path = os.path.join(task_dir, "video")
            os.makedirs(video_path, exist_ok=True)
            for key in video_dict:
                if video_dict[key] != None:
                    imageio.mimsave(os.path.join(video_path, f"{key}_video.mp4"), np.stack(video_dict[key]), fps=24, macro_block_size=1)