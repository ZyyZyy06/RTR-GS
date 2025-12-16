import math
import torch
import torch.nn.functional as F
from arguments import OptimizationParams
from pbr.shade import get_reflectance_color, pbr_shading
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from utils.prt_utils import PRTutils
from utils.sh_utils import eval_sh
from utils.loss_utils import ssim, tv_loss, first_order_edge_aware_loss
from utils.image_utils import psnr
from utils.graphics_utils import  linear2srgb_torch
from .rtr_gs_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from gs_ir import recon_occlusion
import nvdiffrast.torch as dr

def render_view(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                scaling_modifier=1.0, override_color=None, is_training=False, dict_params=None):
    
    gamma_func = lambda x : linear2srgb_torch(x)
    refmap = dict_params["refmap"]

    if pc.use_pbr:
        cubemap = dict_params["cubemap"]
    
    if is_training:
        refmap.train()
        refmap.build_mips()
        if pc.use_pbr:
            cubemap.train()
            cubemap.build_mips()
    else:
        refmap.eval()
        if pc.use_pbr:
            cubemap.eval()

    
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    intrinsic = viewpoint_camera.intrinsics

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        cx=float(intrinsic[0, 2]),
        cy=float(intrinsic[1, 2]),
        bg=torch.zeros_like(bg_color),
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        backward_geometry=True,
        computer_pseudo_normal=True,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # set attribuates
    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity
    ref_tint = pc.get_ref_tint
    ref_roughness = pc.get_ref_roughness
    ref_strength = pc.get_ref_strength
    normal = pc.get_min_axis(viewpoint_camera.camera_center)


    dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_shs.shape[0], 1))
    dir_pp_normalized = F.normalize(dir_pp, dim=-1)
    

    xyz_homo = torch.cat([means3D, torch.ones_like(means3D[:, :1])], dim=-1)
    depths = (xyz_homo @ viewpoint_camera.world_view_transform)[:, 2:3]
    depths2 = depths.square()
    
    only_diffuse = dict_params["iteration"] < pipe.diffuse_iteration
    if pipe.compute_with_prt and override_color is None:
        net = dict_params["transfer_net"]
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
        if only_diffuse:
            prt_color = PRTutils.cal_diffuse(pc)
        else:
            prt_color = PRTutils.cal_color(pc, net, viewdirs,  normal, is_training)
        override_color = prt_color
    elif pipe.compute_with_prt and override_color is not None:
        1 / 0



    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.compute_SHs_python:
            dir_pp_normalized = F.normalize(viewpoint_camera.camera_center.repeat(means3D.shape[0], 1) - means3D,
                                            dim=-1)
            shs_view = pc.get_shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_shs
    else:
        colors_precomp = override_color



    features = torch.cat([depths, depths2, normal, ref_tint, ref_roughness, ref_strength], dim=-1) # [1, 1, 3, 3, 1, 1]
    
    if pc.use_pbr:
        base_color = pc.get_base_color
        roughness = pc.get_roughness
        metallic = pc.get_metallic

        # for editing test
        # roughness = torch.ones_like(roughness) - 0.4
        # metallic = torch.ones_like(metallic) - 0.2
        # base_color = base_color[:, [2, 1, 0]]

        # roughness = torch.where(roughness > 0.2, 0, 0.4)

        # r_channel = torch.clamp(base_color[:, 0] + 0.7, 0.0, 1.0)
        # g_channel = torch.clamp(base_color[:, 1] + 0.7, 0.0, 1.0)
        # b_channel = torch.clamp(base_color[:, 1] + 0.7, 0.0, 1.0)
        # base_color[:, 0] = r_channel
        # base_color[:, 1] = g_channel
        # base_color[:, 2] = b_channel
        # end


        incidents = pc.get_incidents  # incident shs
        viewdirs = F.normalize(viewpoint_camera.camera_center - means3D, dim=-1)
        incidents_light = torch.clamp(eval_sh(pc.active_sh_degree, incidents.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2), normal), 0.0, 1.0)

        features = torch.cat([features, base_color, roughness, metallic, incidents_light], dim=-1) # [1, 1, 3, 3, 1, 1, 3, 1, 1, 3]


    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    (num_rendered, num_contrib, rendered_image, rendered_opacity, rendered_depth,
     rendered_feature, rendered_pseudo_normal, rendered_surface_xyz, weights, radii) = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
        features=features,
    )

    # FOR DEFFERED SHADING
    mask = num_contrib > 0
    rendered_feature = rendered_feature / rendered_opacity.clamp_min(1e-5) * mask   #[N, H, W]
    feature_size = rendered_feature.shape[0]



    rendered_depth, rendered_depth2, rendered_normal, rendered_ref_tint, rendered_ref_roughness, rendered_ref_strength_map, rendered_feature_rest \
        = rendered_feature.split([1, 1, 3, 3, 1, 1, feature_size - 10], dim=0)
    if pc.use_pbr:
        rendered_base_color, rendered_roughness, rendered_metallic, rendered_incident_lights, rendered_feature_rest_2 \
            = rendered_feature_rest.split([3, 1, 1, 3, feature_size - 18], dim=0)


    rendered_var = rendered_depth2 - rendered_depth.square()    # [1, H, W]


    # Radiance Shading
    depth_map = rendered_depth.permute(1, 2, 0)
    opacity_map = rendered_opacity.permute(1, 2, 0)                                         # [H, W, 1]
    
    ref_roughness_map = rendered_ref_roughness.permute(1, 2, 0)                             # [H, W, 1]
    ref_tint_map = rendered_ref_tint.permute(1, 2, 0)                                       # [H, W, 3]
    ref_strength_map = rendered_ref_strength_map.permute(1, 2, 0)                           # [H, W, 1]
    normal_map = rendered_normal.permute(1, 2, 0)                                           # [H, W, 3]
    normal_map = F.normalize(normal_map, dim=-1)
    radiance_map = rendered_image.permute(1, 2, 0)                                          # [H, W, 3]


    canonical_rays = dict_params["canonical_rays"]
    c2w = viewpoint_camera.c2w
    H, W = viewpoint_camera.image_height, viewpoint_camera.image_width

    view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]
    

    refl_color =  get_reflectance_color(refmap, normal_map, view_dirs, ref_roughness_map, ref_tint_map, brdf_lut=dict_params["brdf_lut"])
    

    ref_rgb = (1.0 - ref_strength_map) * radiance_map + ref_strength_map * refl_color
    ref_rgb = ref_rgb * opacity_map + (1.0 - opacity_map) * bg_color

    out_feature_dict = {}
    out_feature_dict.update({
        "ref_roughness": ref_roughness_map.permute(2, 0, 1),    # [1, H, W]
        "ref_strength": ref_strength_map.permute(2, 0, 1),      # [1, H, W]
    })


    #PBR SHADING
    if pc.use_pbr:
        roughness_map = rendered_roughness.permute(1, 2, 0)                                 # [H, W, 1]
        roughness_map = torch.clamp(roughness_map, 0.04, 1.0)                               # [H, W, 1]
        metallic_map = rendered_metallic.permute(1, 2, 0)                                   # [H, W, 1]
        base_color_map = rendered_base_color.permute(1, 2, 0)                               # [H, W, 3]

        incident_light_map = rendered_incident_lights.permute(1, 2, 0)                      # [H, W, 3]

        # get occulsion
        points = (
            (-view_dirs.reshape(-1, 3) * rendered_depth.reshape(-1, 1) + c2w[:3, 3])
                .clamp(min=-1.5, max=1.5)
                    .contiguous()
                )  # [HW, 3]
        
        if "occlusion_volumes" in dict_params.keys():
            occlusion_volumes = dict_params["occlusion_volumes"]
            aabb = dict_params["aabb"]
            occlusion_map = recon_occlusion(
                            H=H,
                            W=W,
                            bound = occlusion_volumes["bound"],
                            points = points,
                            normals = normal_map.reshape(-1, 3).contiguous(),
                            roughness = roughness_map.reshape(-1, 1).contiguous(),
                            occlusion_coefficients = occlusion_volumes["occlusion_coefficients"],
                            occlusion_ids= occlusion_volumes["occlusion_ids"],
                            aabb = aabb,
                            degree = occlusion_volumes["degree"],
                        ).reshape(H, W, 1)
        else:
            occlusion_map = None

        

        pbr_result = pbr_shading(
            light=cubemap,
            normals = normal_map,  # [H, W, 3]
            view_dirs = view_dirs,   # [H, W, 3]
            albedo = base_color_map,  # [H, W, 3]
            roughness = roughness_map,  # [H, W, 1]
            metallic = metallic_map if pipe.metallic else None,    # [H, W, 1]
            occlusion = occlusion_map if occlusion_map is not None else None,  # [H, W, 1]
            irradiance = incident_light_map if not pipe.relight else None,     # [H, W, 3]
            brdf_lut=dict_params["brdf_lut"],
        )

        rendered_pbr = pbr_result["render_rgb"] # [H, W, 3]

        diffuse_pbr = pbr_result["diffuse_rgb"] # [H, W, 3]
        specular_pbr = pbr_result["specular_rgb"] # [H, W, 3]
        occulusion_incident_light = pbr_result["incidents_light"] # [H, W, 1]


        rendered_pbr = rendered_pbr * opacity_map + (1.0 - opacity_map) * bg_color

        if pipe.tone_mapping:
            rendered_pbr = torch.clamp(rendered_pbr, 0.0, 1.0)
        

        out_feature_dict.update({
            "base_color": base_color_map.permute(2, 0, 1),
            "roughness": roughness_map.permute(2, 0, 1),
            "metallic": metallic_map.permute(2, 0, 1),
        })


        out_feature_dict.update({
            "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None else torch.zeros_like(roughness_map).permute(2, 0, 1),
        })




    vis_dict = {}
    if not is_training:
        blended_radiance = (1.0 - ref_strength_map) * radiance_map
        blended_ref_color = ref_strength_map * refl_color
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
        vis_dict.update({
                "depth": depth_map.permute(2, 0, 1),
                "normal": (normal_map.permute(2, 0, 1) * 0.5 + 0.5),
                "pseudo_normal" : rendered_pseudo_normal * 0.5 + 0.5,
                "ref_roughness": ref_roughness_map.permute(2, 0, 1),
                "ref_strength": ref_strength_map.permute(2, 0, 1),
                "radiance_color": radiance_map.permute(2, 0, 1),
                "ref_color": refl_color.permute(2, 0, 1),
                "ref_export_base": refmap.export_envmap(return_img=True).permute(2, 0, 1),
                "ref_tint": ref_tint_map.permute(2, 0, 1),
                "blended_radiance": blended_radiance.permute(2, 0, 1),
                "blended_ref_color": blended_ref_color.permute(2, 0, 1)
            }
        )

        if pc.use_pbr:
            vis_dict.update({
                "base_color": gamma_func(base_color_map.permute(2, 0, 1)),
                "base_color_rgb": base_color_map.permute(2, 0, 1),
                "roughness": roughness_map.permute(2, 0, 1),
                "metallic": metallic_map.permute(2, 0, 1),
            })

                
            vis_dict.update({
                    "visibility": occlusion_map.permute(2, 0, 1) if occlusion_map is not None else torch.zeros_like(rendered_image),
                    "diffuse_pbr": gamma_func(diffuse_pbr.permute(2, 0, 1)), 
                    "specular_pbr": gamma_func(specular_pbr.permute(2, 0, 1)),
                    "image_pbr": gamma_func(rendered_pbr.permute(2, 0, 1)),
                    "incidents_light": (occulusion_incident_light.permute(2, 0, 1)),
                })

            vis_dict.update({
                "env_export_base": cubemap.export_envmap(return_img=True).permute(2, 0, 1),
                "env_export_diffuse": cubemap.export_envmap(return_img=True, base=False).permute(2, 0, 1),
            })

        without_opacity_mask_keys = ["env_export_base", "env_export_diffuse", "ref_export_base"] 
        for key in vis_dict.keys():
            if key not in without_opacity_mask_keys:

                vis_dict[key] = (vis_dict[key].permute(1,2,0) * opacity_map + (1.0 - opacity_map) * bg_color).permute(2, 0, 1)
        


        
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    results = {"render": ref_rgb.permute(2, 0, 1),
               "depth": rendered_depth,
               "depth_var": rendered_var,
               "normal": normal_map.permute(2, 0, 1),
               "pseudo_normal": rendered_pseudo_normal,
               "surface_xyz": rendered_surface_xyz,
               "opacity": rendered_opacity,
               "viewspace_points": screenspace_points,
               "visibility_filter": radii > 0,
               "radii": radii,
               "num_rendered": num_rendered,
               "num_contrib": num_contrib,
               "weights": weights
               }
    


    if pc.use_pbr:
        results['pbr'] = gamma_func(rendered_pbr.permute(2, 0, 1))

    results.update(out_feature_dict)
    
    results['vis_dict'] = vis_dict
    
    

    if not is_training:
        if pc.use_pbr:
            directions = viewpoint_camera.get_world_directions()
            directions = directions.permute(1, 2, 0).unsqueeze(0)
            direct_env = dr.texture(
                cubemap.base[None, ...],  # [1, 6, 16, 16, 3]
                directions.contiguous(),  # [1, H, W, 3]
                filter_mode="linear",
                boundary_mode="cube",
            )[0]

            results["pbr_env"] = gamma_func((rendered_pbr * opacity_map + (1 - opacity_map)) * direct_env).permute(2, 0, 1)
            results["env_only"] = gamma_func(direct_env.permute(2, 0, 1))
        

    return results



def calculate_loss(viewpoint_camera, pc, results, opt, env_map):
    tb_dict = {
        "num_points": pc.get_xyz.shape[0],
    }

    rendered_image = results["render"]
    rendered_depth = results["depth"]
    rendered_normal = results["normal"]
    rendered_opacity = results["opacity"]

    rendered_ref_roughness = results["ref_roughness"]
    rendered_ref_strength = results["ref_strength"]

    loss = 0
    gt_image = viewpoint_camera.original_image.cuda()
    Ll1 = F.l1_loss(rendered_image, gt_image)
    ssim_val = ssim(rendered_image, gt_image)
    tb_dict["l1"] = Ll1.item()
    tb_dict["psnr"] = psnr(rendered_image, gt_image).mean().item()
    tb_dict["ssim"] = ssim_val.item()
    loss = opt.lambda_rgb * ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val))


    if opt.lambda_depth > 0:
        gt_depth = viewpoint_camera.depth.cuda()
        image_mask = viewpoint_camera.image_mask.cuda().bool()
        depth_mask = gt_depth > 0
        sur_mask = torch.logical_xor(image_mask, depth_mask)

        loss_depth = F.l1_loss(rendered_depth[~sur_mask], gt_depth[~sur_mask])
        tb_dict["loss_depth"] = loss_depth.item()
        loss = loss + opt.lambda_depth * loss_depth

    if opt.lambda_mask_entropy > 0:
        o = rendered_opacity.clamp(1e-6, 1 - 1e-6)
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_mask_entropy = -(image_mask * torch.log(o) + (1 - image_mask) * torch.log(1 - o)).mean()
        tb_dict["loss_mask_entropy"] = loss_mask_entropy.item()
        loss = loss + opt.lambda_mask_entropy * loss_mask_entropy

    if opt.lambda_normal_render_depth > 0:
        normal_pseudo = results['pseudo_normal']
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_normal_render_depth = F.mse_loss(
            rendered_normal * image_mask, normal_pseudo.detach() * image_mask)
        tb_dict["loss_normal_render_depth"] = loss_normal_render_depth.item()
        loss = loss + opt.lambda_normal_render_depth * loss_normal_render_depth


    if opt.lambda_normal_smooth > 0:
        loss_normal_smooth = tv_loss(rendered_normal * image_mask)
        tb_dict["loss_normal_smooth"] = loss_normal_smooth.item()
        loss = loss + opt.lambda_normal_smooth * loss_normal_smooth

    if opt.lambda_ref_roughness_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_ref_roughness_smooth = first_order_edge_aware_loss(rendered_ref_roughness * image_mask, gt_image)
        tb_dict["loss_ref_roughness_smooth"] = loss_ref_roughness_smooth.item()
        loss = loss + opt.lambda_ref_roughness_smooth * loss_ref_roughness_smooth

    if opt.lambda_ref_strength_smooth > 0:
        image_mask = viewpoint_camera.image_mask.cuda()
        loss_ref_strength_smooth = first_order_edge_aware_loss(rendered_ref_strength * image_mask, gt_image)
        tb_dict["loss_ref_strength_smooth"] = loss_ref_strength_smooth.item()
        loss = loss + opt.lambda_ref_strength_smooth * loss_ref_strength_smooth


    if pc.use_pbr:
        rendered_pbr = results["pbr"]

        Ll1_pbr = F.l1_loss(rendered_pbr, gt_image)
        ssim_val_pbr = ssim(rendered_pbr, gt_image)
        tb_dict["l1_pbr"] = Ll1_pbr.item()
        tb_dict["ssim_pbr"] = ssim_val_pbr.item()
        tb_dict["psnr_pbr"] = psnr(rendered_pbr, gt_image).mean().item()
        loss_pbr = (1.0 - opt.lambda_dssim) * Ll1_pbr + opt.lambda_dssim * (1.0 - ssim_val_pbr)
        loss = loss + opt.lambda_pbr * loss_pbr

        # for metallic roughness workflow
        if opt.lambda_base_color_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_base_color = results["base_color"]
            loss_base_color_smooth = first_order_edge_aware_loss(rendered_base_color * image_mask, gt_image)
            tb_dict["loss_base_color_smooth"] = loss_base_color_smooth.item()
            loss = loss + opt.lambda_base_color_smooth * loss_base_color_smooth

        if opt.lambda_roughness_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_roughness = results["roughness"]
            loss_roughness_smooth = first_order_edge_aware_loss(rendered_roughness * image_mask, gt_image)
            tb_dict["loss_roughness_smooth"] = loss_roughness_smooth.item()
            loss = loss + opt.lambda_roughness_smooth * loss_roughness_smooth

        if opt.lambda_metallic_smooth > 0:
            image_mask = viewpoint_camera.image_mask.cuda()
            rendered_metallic = results["metallic"]
            loss_metallic_smooth = first_order_edge_aware_loss(rendered_metallic * image_mask, gt_image)
            tb_dict["loss_metallic_smooth"] = loss_metallic_smooth.item()
            loss = loss + opt.lambda_metallic_smooth * loss_metallic_smooth


        if opt.lambda_env_smooth > 0:
            env = env_map.get_env_map()
            loss_env_smooth = tv_loss(env.permute(2, 0, 1))
            tb_dict["loss_env_smooth"] = loss_env_smooth
            loss = loss + opt.lambda_env_smooth * loss_env_smooth

        if opt.lambda_white_light > 0:
            env_base = env_map.base
            white = (env_base[..., 0:1] + env_base[..., 1:2] + env_base[..., 2:3]) / 3.0
            loss_light_white_blance = torch.mean(torch.abs(env_base - white))
            tb_dict["loss_light_white_blance"] = loss_light_white_blance.item()
            loss = loss + opt.lambda_white_light * loss_light_white_blance

        if opt.lambda_reflect_strength_equal_metallic > 0:
            loss_reflect_strength_equal_metallic = F.l1_loss(rendered_metallic, rendered_ref_strength)
            tb_dict["loss_reflect_strength_equal_metallic"] = loss_reflect_strength_equal_metallic.item()
            loss = loss + opt.lambda_reflect_strength_equal_metallic * loss_reflect_strength_equal_metallic


    tb_dict["loss"] = loss.item()

    return loss, tb_dict


def render(viewpoint_camera: Camera, pc: GaussianModel, pipe, bg_color: torch.Tensor,
                 scaling_modifier=1.0, override_color=None, opt: OptimizationParams = False,
                 is_training=False, dict_params=None):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    results = render_view(viewpoint_camera, pc, pipe, bg_color,
                          scaling_modifier, override_color, is_training, dict_params)

    if is_training:
        loss, tb_dict = calculate_loss(viewpoint_camera, pc, results, opt, 
                                       env_map=dict_params['cubemap'] if pc.use_pbr else None)
        results["tb_dict"] = tb_dict
        results["loss"] = loss

    return results