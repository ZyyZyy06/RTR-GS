import torch
from scene.gaussian_model import GaussianModel
from scene.transfer_mlp import TransferMLP

class PRTutils:
    
    @staticmethod
    def cal_diffuse(gaussian: GaussianModel, mask = None):

        diffuse_tint = gaussian.get_diffuse_tint.contiguous() if mask is None else gaussian.get_diffuse_tint.contiguous()[mask]
        deg = gaussian.active_sh_degree
        use_len = (deg + 1) ** 2

        shs = gaussian.get_shs if mask is None else gaussian.get_shs[mask]
        shs_direct_light = shs.transpose(1, 2)[..., :use_len]
        shs_transfer = gaussian.get_diffuse_transfer if mask is None else  gaussian.get_diffuse_transfer[mask]
        shs_diffust_transfer = shs_transfer.transpose(1, 2)[..., :use_len]

        transport = torch.relu((shs_diffust_transfer * shs_direct_light).sum(-1) + 0.5)
        cd = (diffuse_tint) * transport

        prt_color = cd

        return prt_color



    @staticmethod
    def cal_specular(gaussian: GaussianModel, net: TransferMLP, dir, normal, mask = None):
        normal_use = normal if mask is None else normal[mask]

        deg = gaussian.active_sh_degree
        use_len = (deg + 1) ** 2

        view_dir = dir
        reflect_dir = 2.0 * (normal_use * view_dir).sum(-1, keepdims=True).clamp(min=0.0) * normal_use - view_dir

        LT_coeff = PRTutils.cal_spec_coff(gaussian, net, reflect_dir, mask).unsqueeze(1)[..., :use_len]
        shs = gaussian.get_shs if mask is None else gaussian.get_shs[mask]
        direct_light_shs = shs.transpose(1, 2).view(-1, 3, (gaussian.max_sh_degree + 1) ** 2)[..., :use_len]

        direct_color = torch.relu((LT_coeff * direct_light_shs).sum(-1))
        specular_tint = gaussian.get_specular_tint if mask is None else gaussian.get_specular_tint[mask]
        cs = specular_tint * direct_color

        return cs

    @staticmethod
    def cal_spec_coff(gaussian: GaussianModel, net: TransferMLP, dir, mask = None):
        spec_feature = gaussian.get_specular_feature if mask is None else gaussian.get_specular_feature[mask]
        # spec_coff = net.forward(torch.cat((spec_feature, specular_tint, dir), dim=-1) )
        spec_coff = net.forward(spec_feature, dir)
        return spec_coff


    @staticmethod
    def cal_color(gaussian: GaussianModel, net: TransferMLP, dir, normal, is_training = False, mask = None):
        diffuse_color = PRTutils.cal_diffuse(gaussian, mask)
        cs = PRTutils.cal_specular(gaussian, net, dir, normal, mask=mask)

        prt_color = diffuse_color + cs
        return prt_color

