import json
import os
import time
from typing import Tuple
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
from skimage import measure
import trimesh
import open3d as o3d

def load_json_config(json_file):
    if not os.path.exists(json_file):
        return None

    with open(json_file, 'r', encoding='UTF-8') as f:
        load_dict = json.load(f)

    return load_dict


@torch.no_grad()
def extract_mesh_unbounded(depthmaps, rgbmaps, viewpoint_stack, points, center, radius, resolution=1024):
    """
    Experimental features, extracting meshes from unbounded scenes, not fully test across datasets. 
    return o3d.mesh
    """
    def contract(x):
        mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
        return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))
    
    def uncontract(y):
        mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
        return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

    def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
        """
            compute per frame sdf
        """
        new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
        z = new_points[..., -1:]
        pix_coords = (new_points[..., :2] / new_points[..., -1:])
        mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
        sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
        sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
        sdf = (sampled_depth-z)
        return sdf, sampled_rgb, mask_proj

    def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
        """
            Fusion all frames, perform adaptive sdf_funcation on the contract spaces.
        """
        if inv_contraction is not None:
            mask = torch.linalg.norm(samples, dim=-1) > 1
            # adaptive sdf_truncation
            sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
            sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
            samples = inv_contraction(samples)
        else:
            sdf_trunc = 5 * voxel_size

        tsdfs = torch.ones_like(samples[:,0]) * 1
        rgbs = torch.zeros((samples.shape[0], 3)).cuda()

        weights = torch.ones_like(samples[:,0])
        for i, viewpoint_cam in tqdm(enumerate(viewpoint_stack), desc="TSDF integration progress"):
            sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                depthmap = depthmaps[i],
                rgbmap = rgbmaps[i],
                viewpoint_cam=viewpoint_stack[i],
            )

            # volume integration
            sdf = sdf.flatten()
            mask_proj = mask_proj & (sdf > -sdf_trunc)
            sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
            w = weights[mask_proj]
            wp = w + 1
            tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
            rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
            # update weight
            weights[mask_proj] = wp
        
        if return_rgb:
            return tsdfs, rgbs

        return tsdfs

    normalize = lambda x: (x - center) / radius
    unnormalize = lambda x: (x * radius) + center
    inv_contraction = lambda x: unnormalize(uncontract(x))

    N = resolution
    voxel_size = (radius * 2 / N)
    print(f"Computing sdf gird resolution {N} x {N} x {N}")
    print(f"Define the voxel_size as {voxel_size}")
    sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
    R = contract(normalize(points)).norm(dim=-1).cpu().numpy()
    R = np.quantile(R, q=0.95)
    R = min(R+0.01, 1.9)

    mesh = marching_cubes_with_contraction(
        sdf=sdf_function,
        bounding_box_min=(-R, -R, -R),
        bounding_box_max=(R, R, R),
        level=0,
        resolution=N,
        inv_contraction=inv_contraction,
    )
    
    # coloring the mesh
    torch.cuda.empty_cache()
    mesh = mesh.as_open3d
    print("texturing mesh ... ")
    _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
    mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())
    return mesh

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt



def estimate_bounding_sphere(viewpoint_stack):
    """
    Estimate the bounding sphere given camera pose
    """
    torch.cuda.empty_cache()
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_stack])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
    center = torch.from_numpy(center).float().cuda()
    print(f"The estimated bounding radius is {radius:.2f}")
    print(f"Use at least {2.0 * radius:.2f} for depth_trunc")
    return center, radius


def marching_cubes_with_contraction(
    sdf,
    resolution=512,
    bounding_box_min=(-1.0, -1.0, -1.0),
    bounding_box_max=(1.0, 1.0, 1.0),
    return_mesh=False,
    level=0,
    simplify_mesh=True,
    inv_contraction=None,
    max_range=32.0,
):
    assert resolution % 512 == 0

    resN = resolution
    cropN = 512
    level = 0
    N = resN // cropN

    grid_min = bounding_box_min
    grid_max = bounding_box_max
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    meshes = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                x = torch.linspace(x_min, x_max, cropN).cuda()
                y = torch.linspace(y_min, y_max, cropN).cuda()
                z = torch.linspace(z_min, z_max, cropN).cuda()

                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 256**3, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # construct point pyramids
                points = points.reshape(cropN, cropN, cropN, 3)
                points = points.reshape(-1, 3)
                pts_sdf = evaluate(points.contiguous())
                z = pts_sdf.detach().cpu().numpy()
                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    verts = verts + np.array([x_min, y_min, z_min])
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)
                
                print("finished one block")

    combined = trimesh.util.concatenate(meshes)
    combined.merge_vertices(digits_vertex=6)

    # inverse contraction and clipping the points range
    if inv_contraction is not None:
        combined.vertices = inv_contraction(torch.from_numpy(combined.vertices).float().cuda()).cpu().numpy()
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)
    
    return combined


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


    editing_name = "default_edit"
    if pipe.editing_config_path != "" and is_pbr:
        if os.path.exists(pipe.editing_config_path):
            with open(pipe.editing_config_path) as json_file:
                editing_config = json.load(json_file)
                editing_name = editing_config["name"]
                gaussians.editing_materials(editing_config["configs"])
        else:
            NotImplementedError("No editing config found")


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



    if not args.skip_render:
        test_transforms_file = os.path.join(args.source_path, "transforms_test.json")
        contents = load_json_config(test_transforms_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        if pipe.editing_config_path != "":
            save_name = args.save_name + "/" + editing_name
        else:
            save_name = args.save_name
        eval_render(gaussians, render_fn, pipe, background, opt, pbr_kwargs, frames, fovx, dataset.white_background, args.skip_save_image, args.skip_eval, save_name, args.save_video)
    if args.save_mesh:
        train_transforms_file = os.path.join(args.source_path, "transforms_train.json")
        contents = load_json_config(train_transforms_file)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]
        exported_mesh(gaussians, render_fn, pipe, background, opt, pbr_kwargs, frames, fovx, dataset.white_background)


def exported_mesh(gaussians, render_fn, pipe, background, opt, pbr_kwargs, frames, fovx, white_background):
    rgbmaps = []
    depthmaps = []
    viewpoint_stack = []

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

            H = image.shape[0]
            W = image.shape[1]
            fovy = focal2fov(fov2focal(fovx, W), H)

            custom_cam = Camera(colmap_id=0, R=R, T=T,
                                FoVx=fovx, FoVy=fovy, fx=None, fy=None, cx=None, cy=None,
                                image=torch.zeros(3, H, W), image_name="test", uid=0)

            results = render_fn(custom_cam, gaussians, pipe, background, opt=opt, is_training=False,
                                dict_params=pbr_kwargs)

            vis_dict = results["vis_dict"]

            rgbmaps.append(vis_dict['radiance_color'].cpu())
            depthmaps.append(vis_dict['surf_depth'].cpu())
            viewpoint_stack.append(custom_cam)

        rgbmaps = torch.stack(rgbmaps, dim=0)
        depthmaps = torch.stack(depthmaps, dim=0)
        
        center, radius = estimate_bounding_sphere(viewpoint_stack)
        mesh = extract_mesh_unbounded(depthmaps, rgbmaps, viewpoint_stack, gaussians.get_xyz, center, radius)
        mesh_dir = os.path.join(args.model_path, 'exported_mesh')
        os.makedirs(mesh_dir, exist_ok=True)
        name = 'fuse_unbounded.ply'
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(mesh_dir, name)))
        # post-process the mesh and save, saving the largest N clusters
        mesh_post = post_process_mesh(mesh, cluster_to_keep=50)
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(mesh_dir, name.replace('.ply', '_post.ply'))))

def eval_render(gaussians, render_fn, pipe, background, opt, pbr_kwargs, frames, fovx, white_background, skip_save, skip_eval, save_name, save_video):
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

    rgbmaps = []
    depthmaps = []
    viewpoint_stack = []

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

            rgbmaps.append(vis_dict['radiance_color'])
            depthmaps.append(vis_dict['depth'])
            viewpoint_stack.append(custom_cam)


            image_radiance = results["render"]
            image_radiance = torch.clamp(image_radiance, 0.0, 1.0)

            if not skip_eval:
                psnr_radiance_test += psnr(image_radiance, gt_image).mean().double()
                ssim_radiance_test += ssim(image_radiance, gt_image).mean().double()
                lpips_radiance_test += LPIPS(image_radiance, gt_image).mean().double()

            if is_pbr:
                image_pbr = results["pbr"]
                image_pbr = torch.clamp(image_pbr, 0.0, 1.0)
                if not skip_eval:
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
                    "blended_ref_color" : vis_dict["blended_ref_color"],
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
    if not skip_eval:
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
    parser.add_argument("--skip_render", action='store_true', default=False)
    parser.add_argument("--skip_eval", action='store_true', default=False)
    parser.add_argument("--save_video", action='store_true', default=False)
    parser.add_argument("--save_mesh", action='store_true', default=False)
    parser.add_argument("--fps", action='store_true', default=False)
    

    args = get_combined_args(parser)
    print(f"Current model path: {args.model_path}")
    print(f"Current rendering type:  {args.type}")
    print("Optimizing " + args.model_path)


    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    is_pbr = args.type in ['render_ref_pbr', 'render_ref_fast']
    evaling(lp.extract(args), op.extract(args), pp.extract(args), is_pbr=is_pbr)
