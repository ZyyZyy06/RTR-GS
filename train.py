import os
import time
import torch
import torch.nn.functional as F
import torchvision
from collections import defaultdict
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render_fn_dict
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from utils.system_utils import prepare_output_and_logger
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from torchvision.utils import save_image
from lpipsPyTorch import get_lpips_model
from pbr import CubemapLight, get_brdf_lut
from scene.transfer_mlp import TransferMLP


start_time = 0
end_time = 0


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, is_pbr=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)

    # for real scenes
    USE_ENV_SCOPE = opt.use_env_scope # False
    if USE_ENV_SCOPE:
        center = [float(c) for c in opt.env_scope_center]
        ENV_CENTER = torch.tensor(center, device='cuda')
        ENV_RADIUS = opt.env_scope_radius
        REFL_MSK_LOSS_W = 0.4

    """
    Setup Gaussians
    """
    gaussians = GaussianModel(dataset.sh_degree, render_type=args.type)
    scene = Scene(dataset, gaussians)
    if args.checkpoint:
        print("Create Gaussians from checkpoint {}".format(args.checkpoint))
        first_iter = gaussians.create_from_ckpt(args.checkpoint, restore_optimizer=True)

    elif scene.loaded_iter:
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud",
                                        "iteration_" + str(scene.loaded_iter),
                                        "point_cloud.ply"))
    else:
        gaussians.create_from_pcd(scene.scene_info.point_cloud, scene.cameras_extent)

    gaussians.training_setup(opt)


    """
    Setup PBR components
    """
    pbr_kwargs = dict()

    if pipe.compute_with_prt:
        gaussians.active_sh_degree = gaussians.max_sh_degree
        transfer_net = TransferMLP(sh_degree=gaussians.max_sh_degree, features_n=gaussians.n_featres)
        if args.checkpoint:
            transfer_net_checkpoint = os.path.dirname(args.checkpoint) + "/transfer_net_" + os.path.basename(args.checkpoint)
            if os.path.exists(transfer_net_checkpoint):
                transfer_net.create_from_ckpt(transfer_net_checkpoint)
                print("Successfully loaded transfer net!")
            else:
                print("Failed to load transfer net!")

        transfer_net.training_setup(opt)
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
                print("Failed to load!")
        cubemap.training_setup(opt)
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
                print("Failed to load!")

        refmap.training_setup(opt, light_type="ref")
        pbr_kwargs["refmap"] = refmap




    """ Prepare render function and bg"""
    render_fn = render_fn_dict[args.type]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    start_time = time.time()

    """ Training """
    viewpoint_stack = None
    ema_dict_for_log = defaultdict(int)
    progress_bar = tqdm(range(first_iter + 1, opt.iterations + 1), desc="Training progress",
                        initial=first_iter, total=opt.iterations)
    
    for iteration in progress_bar:
        gaussians.update_learning_rate(iteration)


        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()

        loss = 0
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == args.debug_from:
            pipe.debug = True

        pbr_kwargs["iteration"] = iteration - first_iter
        render_pkg = render_fn(viewpoint_cam, gaussians, pipe, background,
                               opt=opt, is_training=True, dict_params=pbr_kwargs)

        viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        tb_dict = render_pkg["tb_dict"]
        loss += render_pkg["loss"]


        def get_outside_msk():
            return None if not USE_ENV_SCOPE else \
                torch.sum((gaussians.get_xyz - ENV_CENTER[None])**2, dim=-1) > ENV_RADIUS**2

        #refl_mask_loss
        if USE_ENV_SCOPE and 'refl_strength_map' in render_pkg:
            refls = gaussians.get_ref_strength
            refl_msk_loss = refls[get_outside_msk()].mean()
            loss += REFL_MSK_LOSS_W * refl_msk_loss


        loss.backward()

        with torch.no_grad():

            # Progress bar
            pbar_dict = {"num": gaussians.get_xyz.shape[0]}
            for k in tb_dict:
                if k in ["psnr", "psnr_pbr"]:
                    ema_dict_for_log[k] = 0.4 * tb_dict[k] + 0.6 * ema_dict_for_log[k]
                    pbar_dict[k] = f"{ema_dict_for_log[k]:.{7}f}"
            # if iteration % 10 == 0:
            progress_bar.set_postfix(pbar_dict)

            # Log and save
            training_report(tb_writer, iteration, tb_dict,
                            scene, render_fn, pipe=pipe,
                            bg_color=background, dict_params=pbr_kwargs)

            # densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, 
                                                    render_pkg['weights'])
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                        radii[visibility_filter])
                
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)


                
                HAS_RESET0 = False
                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    HAS_RESET0 = True
                    outside_msk = get_outside_msk()
                    gaussians.reset_opacity()
                    if not opt.without_normal_propagation:
                        gaussians.reset_refl(exclusive_msk=outside_msk)

                if  (opt.init_iter < iteration <= opt.propagation_until_iter) and iteration % 1000 == 0 and pipe.ref_map:
                    if not HAS_RESET0 and not opt.without_normal_propagation:
                        outside_msk = get_outside_msk()
                        gaussians.reset_opacity1(exclusive_msk=outside_msk)
                        gaussians.reset_scale(exclusive_msk=outside_msk)

                    
            # Optimizer step
            gaussians.step()
            for component in pbr_kwargs.values():
                try:
                    component.step()
                except:
                    pass
            
            # save checkpoints
            if iteration % args.save_interval == 0 or iteration == args.iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration % args.checkpoint_interval == 0 or iteration == args.iterations:
                os.makedirs(os.path.join(scene.model_path, "checkpoint"),exist_ok=True)
                torch.save((gaussians.capture(), iteration),
                           os.path.join(scene.model_path, "checkpoint/chkpnt" + str(iteration) + ".pth"))

                for com_name, component in pbr_kwargs.items():
                    try:
                        torch.save((component.capture(), iteration),
                                   os.path.join(scene.model_path, f"checkpoint/{com_name}_chkpnt" + str(iteration) + ".pth"))
                        print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    except:
                        pass

                    print("[ITER {}] Saving {} Checkpoint".format(iteration, com_name))

    end_time = time.time()
    with open(os.path.join(args.model_path, "trainint_time.txt"), "w") as f:
        f.write(f"training time(seconds): {end_time - start_time}\n")
        minutes = (end_time - start_time) / 60.0
        f.write(f"training time(minutes): {minutes}\n")


    if dataset.eval and not args.skip_eval:
        eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs)

    


def training_report(tb_writer, iteration, tb_dict, scene: Scene, renderFunc, pipe,
                    bg_color: torch.Tensor, scaling_modifier=1.0, override_color=None,
                    opt: OptimizationParams = None, is_training=False, **kwargs):
    if tb_writer:
        for key in tb_dict:
            tb_writer.add_scalar(f'train_loss_patches/{key}', tb_dict[key], iteration)

    # Report test and samples of training set
    if iteration % args.test_interval == 0:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': scene.getTrainCameras()})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0

                if scene.gaussians.use_pbr:
                    l1_pbr_test = 0.0
                    psnr_pbr_test = 0.0

                for idx, viewpoint in enumerate(
                        tqdm(config['cameras'], desc="Evaluating " + config['name'], leave=False)):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, pipe, bg_color,
                                            scaling_modifier, override_color, opt, is_training,
                                            **kwargs)

                    write_image_dict = {}

                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = viewpoint.original_image.cuda()

                    opacity = torch.clamp(render_pkg["opacity"], 0.0, 1.0)
                    depth = render_pkg["depth"]
                    depth = (depth - depth.min()) / (depth.max() - depth.min())

                    
                    write_image_dict.update({
                        "image": image, "gt_image": gt_image,
                        "opacity": opacity, "depth": depth, 
                    })
                    

                    vis_dict = render_pkg["vis_dict"]
                    write_image_dict.update(vis_dict)

                    
                    if tb_writer and (idx < 10):
                        for key in write_image_dict:
                            tb_writer.add_images(config['name'] + "_view_{}_{}/{}".format(viewpoint.image_name, idx, key),
                                                torch.clamp(write_image_dict[key][None], 0.0, 1.0), global_step=iteration)

                    l1_test += F.l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    if scene.gaussians.use_pbr:
                        l1_pbr_test += F.l1_loss(render_pkg["pbr"], gt_image).mean().double()
                        psnr_pbr_test += psnr(render_pkg["pbr"], gt_image).mean().double()


                psnr_test /= len(config['cameras'])

                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test,
                                                                                    psnr_test))

                if scene.gaussians.use_pbr:
                    psnr_pbr_test /= len(config['cameras'])
                    l1_pbr_test /= len(config['cameras'])
                    print("\n[ITER {}] Evaluating {}: L1 {} PSNR_PBR {}".format(iteration, config['name'], l1_pbr_test,
                                                                                psnr_pbr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

                    if scene.gaussians.use_pbr:
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss_pbr', l1_pbr_test, iteration)
                        tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr_pbr', psnr_pbr_test, iteration)
                if iteration == args.iterations:
                    with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                        f.write("L1 {} PSNR {}".format(l1_test, psnr_test))
                    if scene.gaussians.use_pbr:
                        with open(os.path.join(args.model_path, config['name'] + "_loss.txt"), 'w') as f:
                            f.write("L1 {} PSNR {} PSNR_PBR {}".format(l1_test, psnr_test, psnr_pbr_test))
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()




def eval_render(scene, gaussians, render_fn, pipe, background, opt, pbr_kwargs):
    LPIPS = get_lpips_model(net_type='vgg').cuda()

    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0
    test_cameras = scene.getTestCameras()

    mkdir_flag = False

    if gaussians.use_pbr:
        psnr_pbr_test = 0.0
        ssim_pbr_test = 0.0
        lpips_pbr_test = 0.0
        
        os.makedirs(os.path.join(args.model_path, 'eval'), exist_ok=True)
        env_cubemap = pbr_kwargs['cubemap']
        envmap = env_cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
        envmap_path = os.path.join(args.model_path, 'eval', 'envmap.png')
        torchvision.utils.save_image(envmap, envmap_path)


    progress_bar = tqdm(range(0, len(test_cameras)), desc="Evaluating",
                        initial=0, total=len(test_cameras))

    with torch.no_grad():
        for idx in progress_bar:
            viewpoint = test_cameras[idx]
            results = render_fn(viewpoint, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)


            image = results["render"]
            image = torch.clamp(image, 0.0, 1.0)
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            psnr_test += psnr(image, gt_image).mean().double()
            ssim_test += ssim(image, gt_image).mean().double()
            lpips_test += LPIPS(image, gt_image).mean().double()

            if gaussians.use_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)

                psnr_pbr_test += psnr(image_pbr, gt_image).mean().double()
                ssim_pbr_test += ssim(image_pbr, gt_image).mean().double()
                lpips_pbr_test += LPIPS(image_pbr, gt_image).mean().double()

                
            write_image_dict = {}
            write_image_dict.update({
                "render": image, 
                "gt": gt_image, 
            })

            vis_dict = results["vis_dict"]  
            write_image_dict.update(vis_dict)
            ban_image_keys = ["env_export_base", "env_export_diffuse", "base_color", "ref_export_base", "ref_tint",\
                              "radiance_color", "ref_roughness", "ref_strength", "ref_color"]
            # ban_image_keys = []

            if not mkdir_flag:
                mkdir_flag = True
                os.makedirs(os.path.join(args.model_path, 'eval', 'render'), exist_ok=True)
                os.makedirs(os.path.join(args.model_path, 'eval', 'gt'), exist_ok=True)
                os.makedirs(os.path.join(args.model_path, 'eval', 'normal'), exist_ok=True)
                for key in vis_dict:
                    if key in write_image_dict.keys() and key not in ban_image_keys:
                        os.makedirs(os.path.join(args.model_path, 'eval', key), exist_ok=True)

            for key in write_image_dict:
                if key not in ban_image_keys:
                    save_image(torch.clamp(write_image_dict[key], 0.0, 1.0), 
                            os.path.join(args.model_path, 'eval', key, f"{viewpoint.image_name}_{idx}.png"))


    psnr_test /= len(test_cameras)
    ssim_test /= len(test_cameras)
    lpips_test /= len(test_cameras)

    if gaussians.use_pbr:
        psnr_pbr_test /= len(test_cameras)
        ssim_pbr_test  /= len(test_cameras)
        lpips_pbr_test /= len(test_cameras)

    with open(os.path.join(args.model_path, 'eval', "eval.txt"), "w") as f:
        f.write(f"psnr: {psnr_test}\n")
        f.write(f"ssim: {ssim_test}\n")
        f.write(f"lpips: {lpips_test}\n")

        if gaussians.use_pbr:
            f.write(f"psnr_pbr: {psnr_pbr_test}\n")
            f.write(f"ssim_pbr: {ssim_pbr_test}\n")
            f.write(f"lpips_pbr: {lpips_pbr_test}\n")
    
    if gaussians.use_pbr:
        print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {} PSNR_pbr {} SSIM_pbr {} LPIPS_pbr {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test,  psnr_pbr_test, ssim_pbr_test, lpips_pbr_test))
    else:
        print("\n[ITER {}] Evaluating {}: PSNR {} SSIM {} LPIPS {}".format(args.iterations, "test", psnr_test, ssim_test,
                                                                       lpips_test))


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
    parser.add_argument("--test_interval", type=int, default=4000)
    parser.add_argument("--save_interval", type=int, default=30000)
    parser.add_argument("--skip_eval", action="store_true", default=False)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_interval", type=int, default=30000)
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    parser.add_argument("--occlusion_path", type=str, default=None)

    args = parser.parse_args(sys.argv[1:])
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['neilf_ref_pbr']
    training(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)

    # All done
    print("\nTraining complete.")
