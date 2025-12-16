import json
import os
import time
import imageio
import numpy as np
import torch
import torchvision
from scene.cameras import Camera
from scene.utils import load_img_rgb
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from lpipsPyTorch import get_lpips_model
from pbr import CubemapLight, get_brdf_lut
from scene.transfer_mlp import TransferMLP
from arguments import ModelParams, PipelineParams, get_combined_args



def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


def evaling(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    else:
        NotImplementedError("No checkpoint or loaded iteration found")

    """
    Setup PBR components
    """
    pbr_kwargs = dict()
    pbr_kwargs["iteration"] = first_iter

    if pipe.compute_with_prt:
        transfer_net = TransferMLP(sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres)
        if args.checkpoint:
            transfer_net_checkpoint = os.path.dirname(args.checkpoint) + "/transfer_net_" + os.path.basename(args.checkpoint)
            if os.path.exists(transfer_net_checkpoint):
                transfer_net.create_from_ckpt(transfer_net_checkpoint)
                print("Successfully loaded transfer net!")
            else:
                 NotImplementedError("No checkpoint or loaded iteration found")
        pbr_kwargs["transfer_net"] = transfer_net


    if is_pbr or pipe.ref_map:
        canonical_rays = scene.get_canonical_rays()
        pbr_kwargs["canonical_rays"] = canonical_rays

        brdf_lut = get_brdf_lut().cuda()
        pbr_kwargs["brdf_lut"] = brdf_lut

    if is_pbr:
        if args.occlusion_path is not None:
            occlusion_volumes = torch.load(args.occlusion_path)
            bound = occlusion_volumes["bound"]
            aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
            pbr_kwargs["occlusion_volumes"] = occlusion_volumes
            pbr_kwargs["aabb"] = aabb

        cubemap = CubemapLight(base_res=128).cuda()
        if args.checkpoint:
            cubemap_checkpoint = os.path.dirname(args.checkpoint) + "/cubemap_" + os.path.basename(args.checkpoint)
            if os.path.exists(cubemap_checkpoint):
                cubemap.create_from_ckpt(cubemap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
        else:
            NotImplementedError("No checkpoint or loaded iteration found")
        cubemap.build_mips()
        pbr_kwargs["cubemap"] = cubemap
        
        
    if pipe.ref_map:
        refmap = CubemapLight(base_res=128).cuda()
    
        if args.checkpoint:
            refmap_checkpoint = os.path.dirname(args.checkpoint) + "/refmap_" + os.path.basename(args.checkpoint)
            if os.path.exists(refmap_checkpoint):
                refmap.create_from_ckpt(refmap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
        else:
            NotImplementedError("No checkpoint or loaded iteration found")

        refmap.build_mips()
        pbr_kwargs["refmap"] = refmap

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
    contents = load_json_config(test_transforms_file)
    fovx = contents["camera_angle_x"]
    frames = contents["frames"]

    eval_render(gaussians, render_fn, pipe, background, opt, pbr_kwargs, frames, fovx, dataset.white_background, args.skip_save_image, args.save_name, args.save_video)





def eval_render(gaussians, render_fn, pipe, background, opt, pbr_kwargs, frames, fovx, white_background, skip_save, save_name, save_video):
    LPIPS = get_lpips_model(net_type='vgg').cuda()

    psnr_radiance_test = 0.0
    ssim_radiance_test = 0.0
    lpips_radiance_test = 0.0

    psnr_pbr_test = 0.0
    ssim_pbr_test = 0.0
    lpips_pbr_test = 0.0

    render_time = 0.0

    mkdir_flag = False


    bg = 1 if white_background else 0


    os.makedirs(os.path.join(args.model_path, save_name), exist_ok=True)
    
    if is_pbr:
        env_cubemap = pbr_kwargs['cubemap']
        envmap = env_cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, save_name, 'envmap.png')
        torchvision.utils.save_image(envmap, envmap_path)


    video_dict = {}

    with torch.no_grad():
        for idx, frame in enumerate(tqdm(frames, leave=False)):
            image_path = os.path.join(args.source_path, frame["file_path"] + ".png")

            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            
            image_rgba = load_img_rgb(image_path)
            image = image_rgba[..., :3]
            mask = image_rgba[..., 3:]

            gt_image = torch.from_numpy(image).permute(2, 0, 1).float().cuda()
            mask = torch.from_numpy(mask).permute(2, 0, 1).float().cuda()
            gt_image = gt_image * mask + bg * (1 - mask)

            H = image.shape[0]
            W = image.shape[1]
            fovy = focal2fov(fov2focal(fovx, W), H)

            custom_cam = Camera(colmap_id=0, R=R, T=T,
                                FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                                image=torch.zeros(3, H, W), image_name="test", uid=0)

            if idx > 0:
                t1 = time.time()
            results = render_fn(custom_cam, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            if idx > 0:
                render_time += time.time() - t1

            vis_dict = results["vis_dict"]


            image_radiance = results["render"]
            image_radiance = torch.clamp(image_radiance, 0.0, 1.0)

            psnr_radiance_test += psnr(image_radiance, gt_image).mean().double()
            ssim_radiance_test += ssim(image_radiance, gt_image).mean().double()
            lpips_radiance_test += LPIPS(image_radiance, gt_image).mean().double()

            if is_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)
                psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()
                ssim_pbr_test += ssim(image_pbr, gt_image).mean().double()
                lpips_pbr_test += LPIPS(image_pbr, gt_image).mean().double()



            if not skip_save:
                write_image_dict = {}
                write_image_dict.update({
                    "gt": gt_image, 
                    "render_radiance" :image_radiance,
                })

                if is_pbr:
                    write_image_dict.update({
                        "render_pbr": image_pbr,
                        "incidents_light": vis_dict["incidents_light"],
                        "visibility": vis_dict["visibility"],
                    })

                write_image_dict.update({
                    "depth": vis_dict["depth"],
                    "normal": vis_dict["normal"],
                    "pseudo_normal": vis_dict["pseudo_normal"],

                    "radiance_color": vis_dict["radiance_color"],
                    "ref_color": vis_dict["ref_color"],
                    "ref_strength": vis_dict["ref_strength"],
                    "ref_roughness": vis_dict["ref_roughness"],
                    "blended_radiance" : vis_dict["blended_radiance"],
                    "blended_ref_color" : vis_dict["blended_ref_color"]
                })

                if is_pbr:
                    write_image_dict.update({
                        "roughness": vis_dict["roughness"],
                        "metallic": vis_dict["metallic"],
                        "albedo": vis_dict["base_color_rgb"],
                        "diffuse_pbr": vis_dict["diffuse_pbr"],
                        "specular_pbr": vis_dict["specular_pbr"],
                    })


                if not mkdir_flag:
                    mkdir_flag = True
                    
                    os.makedirs(os.path.join(args.model_path, save_name, 'render_radiance'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'gt'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'normal'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'depth'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'pseudo_normal'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'visibility'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'incidents_light'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'radiance_color'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'ref_color'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'ref_roughness'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'ref_strength'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'blended_radiance'), exist_ok=True)
                    os.makedirs(os.path.join(args.model_path, save_name, 'blended_ref_color'), exist_ok=True)

                    if is_pbr:
                        os.makedirs(os.path.join(args.model_path, save_name, 'render_pbr'), exist_ok=True)
                        os.makedirs(os.path.join(args.model_path, save_name, 'albedo'), exist_ok=True)
                        os.makedirs(os.path.join(args.model_path, save_name, 'metallic'), exist_ok=True)
                        os.makedirs(os.path.join(args.model_path, save_name, 'roughness'), exist_ok=True)
                        os.makedirs(os.path.join(args.model_path, save_name, 'diffuse_pbr'), exist_ok=True)
                        os.makedirs(os.path.join(args.model_path, save_name, 'specular_pbr'), exist_ok=True)


                    for key in write_image_dict:
                        video_dict[key] = []

                for key in write_image_dict:
                        if write_image_dict[key] != None:
                            save_image(torch.clamp(write_image_dict[key], 0.0, 1.0), 
                                os.path.join(args.model_path, save_name, key, f"{custom_cam.image_name}_{idx}.png"))

                            video_image = torch.clamp(write_image_dict[key], 0.0, 1.0).permute(1,2,0).detach().cpu()
                            video_image = (video_image.numpy() * 255).astype('uint8')
                            video_dict[key].append(video_image)


    psnr_radiance_test /= len(frames)
    ssim_radiance_test /= len(frames)
    lpips_radiance_test /= len(frames)

    psnr_pbr_test /= len(frames)
    ssim_pbr_test  /= len(frames)
    lpips_pbr_test /= len(frames)

    fps = 1.0/ (render_time / (len(frames)-1))

    print("fps: ", fps)

    with open(os.path.join(args.model_path, "metrics_{}.txt".format(save_name)), "w") as f:
        f.write(f"psnr_radiance: {psnr_radiance_test}\n")
        f.write(f"ssim_radiance: {ssim_radiance_test}\n")
        f.write(f"lpips_radiance: {lpips_radiance_test}\n")

        f.write(f"psnr_pbr: {psnr_pbr_test}\n")
        f.write(f"ssim_pbr: {ssim_pbr_test}\n")
        f.write(f"lpips_pbr: {lpips_pbr_test}\n")

    print("\n Evaluating {}: PSNR_radiance {} SSIM_radiance {} LPIPS_radiance {}".format("test", psnr_radiance_test, ssim_radiance_test, lpips_radiance_test))
    print("\n Evaluating {}: PSNR_pbr {} SSIM_pbr {} LPIPS_pbr {}".format("test", psnr_pbr_test, ssim_pbr_test, lpips_pbr_test))


    if save_video:
        video_path = os.path.join(args.model_path, save_name, "video")
        os.makedirs(video_path, exist_ok=True)
        for key in video_dict:
            imageio.mimsave(os.path.join(video_path, f"{key}_video.mp4"), np.stack(video_dict[key]), fps=24, macro_block_size=1)

        






if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('-t', '--type', choices=['render_ref', 'render_ref_pbr', 'render_ref_fast'], default='render_ref')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--occlusion_path", type=str, default=None)
    parser.add_argument("--save_name", type=str, default="eval_test")
    parser.add_argument("--skip_save_image", action='store_true', default=False)
    parser.add_argument("--save_video", action='store_true', default=False)
    parser.add_argument("--fps", action='store_true', default=False)

    args = get_combined_args(parser)
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf', 'neilf_ref_pbr', 'neilf_ref_fast']
    evaling(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)
