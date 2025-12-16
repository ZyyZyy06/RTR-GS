from typing import List, Optional

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

from arguments import OptimizationParams
from utils.sh_utils import eval_sh_basis

from .renderutils import diffuse_cubemap, specular_cubemap


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap: torch.Tensor) -> torch.Tensor:
        # avg_pool_nhwc
        y = cubemap.permute(0, 3, 1, 2)  # NHWC -> NCHW
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> torch.Tensor:
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                indexing="ij",
            )
            v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )
        return out


class CubemapLight(nn.Module):
    # for nvdiffrec
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(
        self,
        base_res: int = 512,
        scale: float = 0.5,
        bias: float = 0.25,
    ) -> None:
        super(CubemapLight, self).__init__()
        self.mtx = None
        base = (
            torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale + bias
        )
        # base = (
        #     torch.ones(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale + bias
        # )
        self.base = nn.Parameter(base)
        self.register_parameter("env_base", self.base)
        self.envmap_dirs = self.get_envmap_dirs()
        self.sh_dirs = self.get_sh_dirs()

    def training_setup(self, training_args: OptimizationParams, light_type = "env"):
        assert light_type in ["env", "ref"]
        if light_type == "env":
            lr = training_args.env_lr
        elif light_type == "ref":
            lr = training_args.ref_lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
    def step(self):
        self.base.grad *= 64
        self.optimizer.step()
        
        self.optimizer.zero_grad()
        self.clamp_(min=0.0)

    def xfm(self, mtx) -> None:
        self.mtx = mtx

    def clamp_(self, min: Optional[float]=None, max: Optional[float]=None) -> None:
        self.base.clamp_(min, max)

    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS)
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / (1.0 - self.MAX_ROUGHNESS)
            + len(self.specular)
            - 2,
        )

    def build_sh(self, degree: int = 1):

        num = 100
        sx = torch.rand(num)
        sy = torch.rand(num)
        # phi = torch.arccos(1.0 - 2 * sx) - np.pi / 2.0
        # theta = 2 * np.pi * sy - np.pi
        theta = (2.0 * torch.acos(torch.sqrt(1.0 - sx))).cuda()
        phi = (2.0 * torch.pi * sy).cuda()

        dirs = torch.stack([ torch.sin(theta) * torch.cos(phi), 
                torch.sin(theta) * torch.sin(phi), 
                torch.cos(theta)], dim=-1).view(-1, 3)


        rgbs = dr.texture(
                self.base[None, ...],
                dirs[None,None,...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
        self.shs = eval_sh_basis(degree, dirs, rgbs)


        
    
    def build_mips(self, cutoff: float = 0.99) -> None:
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        self.diffuse = diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (
                self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
            ) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

    def export_envmap(
        self,
        filename: Optional[str] = None,
        res: List[int] = [512, 1024],
        return_img: bool = False,
        base: bool = True,
    ) -> Optional[torch.Tensor]:
        lat_step_size = np.pi / res[0]
        lng_step_size = 2 * np.pi / res[1]
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, res[0],device="cuda"), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, res[1],device="cuda"  )], indexing='ij')


        reflvec = torch.stack([  torch.cos(theta) * torch.cos(phi), 
                                torch.sin(theta) * torch.cos(phi), 
                                torch.sin(phi)], dim=-1).view(res[0], res[1], 3)    # [envH, envW, 3]
        
        if base:
            color = dr.texture(
                self.base[None, ...],
                reflvec[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
        else:
            color = dr.texture(
                self.diffuse[None, ...],
                reflvec[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
        if return_img:
            return color
        else:
            cv2.imwrite(filename, color.clamp(min=0.0).detach().cpu().numpy()[..., ::-1])

    def regularizer(self):
        white = (self.base[..., 0:1] + self.base[..., 1:2] + self.base[..., 2:3]) / 3.0
        return torch.mean(torch.abs(self.base - white))
    
    

    @staticmethod
    def get_sh_dirs():
        pass


    @staticmethod
    def get_envmap_dirs(res: List[int] = [512, 1024]) -> torch.Tensor:
        lat_step_size = np.pi / res[0]
        lng_step_size = 2 * np.pi / res[1]
        phi, theta = torch.meshgrid([torch.linspace(np.pi / 2 - 0.5 * lat_step_size, -np.pi / 2 + 0.5 * lat_step_size, res[0], device="cuda"), 
                                    torch.linspace(np.pi - 0.5 * lng_step_size, -np.pi + 0.5 * lng_step_size, res[1], device="cuda")], indexing='ij')


        view_dirs = torch.stack([  torch.cos(theta) * torch.cos(phi), 
                                torch.sin(theta) * torch.cos(phi), 
                                torch.sin(phi)], dim=-1).view(res[0], res[1], 3)    # [envH, envW, 3]
        
        return view_dirs
    
    def get_env_map(self):
        envmap = dr.texture(
            self.base[None, ...],
            self.envmap_dirs[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        return envmap
    
    
    def capture(self):
        captured_list = [
            self.base,
            self.optimizer.state_dict(),
        ]

        return captured_list
    
    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        (model_args, first_iter) = torch.load(checkpoint_path)
        (self.base,
         opt_dict) = model_args[:2]
        
        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_dict)
            except:
                print("Not loading optimizer state_dict!")

        return first_iter
