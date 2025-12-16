import copy
import os
from typing import List
import numpy as np
import torch
from scene.cameras import Camera
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

from utils.graphics_utils import focal2fov, fov2focal
from torchvision.utils import save_image
from pbr import CubemapLight, get_brdf_lut
from scene.transfer_mlp import TransferMLP
import imageio
from utils.graphics_utils import  read_hdr, latlong_to_cubemap



def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians, shuffle=False)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
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
        cubemap.train()
        if args.checkpoint:
            cubemap_checkpoint = os.path.dirname(args.checkpoint) + "/cubemap_" + os.path.basename(args.checkpoint)
            if os.path.exists(cubemap_checkpoint):
                cubemap.create_from_ckpt(cubemap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                NotImplementedError("No checkpoint or loaded iteration found")
        pbr_kwargs["cubemap"] = cubemap
        
    if pipe.ref_map:
        refmap = CubemapLight(base_res=128).cuda()
        refmap.train()
    
        if args.checkpoint:
            refmap_checkpoint = os.path.dirname(args.checkpoint) + "/refmap_" + os.path.basename(args.checkpoint)
            if os.path.exists(refmap_checkpoint):
                refmap.create_from_ckpt(refmap_checkpoint, restore_optimizer=True)
                print("Successfully loaded!")
            else:
                NotImplementedError("No checkpoint or loaded iteration found")

        refmap.training_setup(opt, light_type="ref")
        pbr_kwargs["refmap"] = refmap

    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not pipe.specular_workflow:
        capture_list = ["pbr", "base_color", "diffuse_pbr", "specular_pbr"]
    else:
        capture_list = ["pbr", "diffuse_color", "specular_color", "diffuse_pbr", "specular_pbr"]

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


    task_names = ['bridge', 'city', 'fireplace', 'forest', 'night']
    for task_name in task_names:
        cubemap = None
        hdri = read_hdr(task_dict[task_name]["envmap_path"])
        hdri = torch.from_numpy(hdri).cuda()
        res = 256
        cubemap = CubemapLight(base_res=res).cuda()
        cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
        cubemap.build_mips()
        cubemap.eval()

        pbr_kwargs["cubemap"] = cubemap

        eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, task_name)

        if args.save_video:
            eval_render_video(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, task_name)
    



def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, env_name, save_video = False):
    test_cameras = scene.getTestCameras()

    mkdir_flag = False
    progress_bar = tqdm(range(0, len(test_cameras)), desc="Relighting",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]


            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            image = results["render"]
            image = torch.clamp(image, 0.0, 1.0)

            if gaussians.use_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)



            write_image_dict = {}
            write_image_dict.update({
                "render": image_pbr, 
            })

            if not mkdir_flag:
                os.makedirs(os.path.join(args.model_path, 'test_rli', env_name), exist_ok=True)
                mkdir_flag = True


            for key in write_image_dict:
                    save_image(torch.clamp(write_image_dict[key], 0.0, 1.0), 
                            os.path.join(args.model_path, 'test_rli', env_name, f"{viewpoint.image_name}_{idx}.png"))
                    

def eval_render_video(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs, env_name):
    test_cameras = scene.getTrainCameras()
    video_images_dict = []
    
    camera = test_cameras[0]
    H = camera.image_height
    W = camera.image_width
    fovx = camera.FoVx
    fovy = focal2fov(fov2focal(fovx, W), H)

    n_frames = 180
    radius = 1  # toycar
    radius = 0.4 #garden
    

    cycle_cameras = []
    def circular_poses(viewpoint_cam, radius, angle=0.0):
        translate_x = radius * np.cos(angle)
        translate_y = radius * np.sin(angle)
        translate_z = 0
        translate = np.array([translate_x, translate_y, translate_z])
        
        custom_cam = Camera(colmap_id=0, R=viewpoint_cam.R, T=viewpoint_cam.T,
            FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
            image=torch.zeros(3, H, W), image_name=None, uid=0,
            trans=translate
        )
        return custom_cam

    for idx in range(n_frames):
        # view = copy.deepcopy(test_cameras[25]) # toycar
        # view = copy.deepcopy(test_cameras[120]) # kitchen
        # view = copy.deepcopy(test_cameras[180]) # kitchen

        view = copy.deepcopy(test_cameras[160]) # garden

        angle = 2 * np.pi * idx / n_frames
        cam = circular_poses(view, radius, angle)
        cycle_cameras.append(cam)
    
    test_cameras = cycle_cameras



    # n_test = 180
    # R_list = []
    # T_list = []

    # fs = [0,  len(test_cameras) - 1, len(test_cameras) // 2, 0]
    # R = test_cameras[fs[0]].R
    # t = test_cameras[fs[0]].T
    # Rt = getWorld2View(R,t)
    # pose0 = Rt
    # for i in range(1, len(fs)):
    #     R = test_cameras[fs[i]].R
    #     t = test_cameras[fs[i]].T
    #     pose1 = getWorld2View(R,t)
    #     rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
    #     slerp = Slerp([0, 1], rots)
    #     for i in range(n_test + 1):
    #         ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
    #         pose = np.eye(4, dtype=np.float32)
    #         pose[:3, :3] = slerp(ratio).as_matrix()
    #         pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
            
    #         R = np.transpose(pose[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
    #         T = pose[:3, 3]

    #         R_list.append(R)
    #         T_list.append(T)
            
    #         # custom_cam = Camera(colmap_id=0, R=R, T=T,
    #         #         FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
    #         #         image=torch.zeros(3, H, W), image_name=None, uid=0)
    #         # trace_cameras.append(custom_cam)
    #     pose0 = pose1


    # progress_bar = tqdm(range(0, len(R_list)), desc="Relighting",
    #                     initial=0, total=len(test_cameras))

    progress_bar = tqdm(range(0, len(test_cameras)), desc="Relighting",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            # custom_cam = Camera(colmap_id=0, R=R_list[idx], T=T_list[idx],
            #         FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
            #         image=torch.zeros(3, H, W), image_name=None, uid=0)
            
            # viewpoint = custom_cam

            viewpoint = test_cameras[idx]


            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)
            image = results["render"]
            image = torch.clamp(image, 0.0, 1.0)

            if gaussians.use_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)

            H, W = image_pbr.shape[1], image_pbr.shape[2]
            H_resize, W_resize = H, W
            if H % 2 != 0:
                H_resize = H - 1
            if W % 2 != 0:
                W_resize = W -1
            # print(H_resize, W_resize)
            # print(H, W)
            # print(H % 2 != 0)

            tmp_image_pbr = image_pbr[:,:H_resize, :W_resize]
            video_image_pbr = torch.clamp(tmp_image_pbr, 0.0, 1.0).permute(1,2,0).detach().cpu()
            video_image_pbr = (video_image_pbr.numpy() * 255).astype('uint8')
            video_images_dict.append(video_image_pbr)



                    
        video_path = os.path.join(args.model_path, 'test_rli', "video")
        os.makedirs(video_path, exist_ok=True)
        imageio.mimsave(os.path.join(video_path, f"{env_name}_pbr_video.mp4"), np.stack(video_images_dict), fps=24, macro_block_size=1)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--gui', action='store_true', default=False, help="use gui")
    parser.add_argument('-t', '--type', choices=['render_ref', 'render_ref_pbr', 'render_ref_fast'], default='render_ref')
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--occlusion_path", type=str, default=None)
    parser.add_argument('-e', '--envmap_path', default="/data/zhouyongyang/dataset/tensorIR/env_maps/high_res_envmaps_1k/", help="Env map path")
    parser.add_argument("--save_video", action="store_true", default=False)

    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf', 'neilf_blend', 'neilf_forward', 'neilf_ref_pbr']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)

    # All done
    print("\nTraining complete.")
