from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, nargs="+")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
                
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, nargs="+")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        self.debug_cuda = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.save_training_vis = False
        self.compute_with_prt = False
        self.deferred_shading = False
        self.diffuse_iteration = 0
        self.tone_mapping = True
        self.metallic = False
        self.ref_map = False
        self.relight = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000

        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.sh_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.ref_lr = 0.0025
        self.env_lr = 0.005

        self.diffuse_tint_lr = 0.01
        self.specular_tint_lr = 0.01
        self.specular_feature_lr = 0.01
        self.diffuse_transfer_lr = 0.01
        self.ref_tint_lr = 0.01
        self.ref_roughness_lr = 0.01
        self.ref_strength_lr = 0.01
        self.mlp_lr = 0.000025

        self.base_color_lr = 0.02
        self.roughness_lr = 0.02
        self.metallic_lr = 0.02

        self.diffuse_color_lr = 0.02
        self.specular_color_lr = 0.02
        self.glossiness_lr = 0.02

        self.light_lr = 0.005
        self.light_rest_lr = 0.005
        self.light_init = 3.0


        self.percent_dense = 0.001
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.init_iter = 3000
        self.densify_until_iter = 10_000
        self.propagation_until_iter = 15_000

        self.densify_grad_threshold = 0.0002

        self.lambda_depth = 0.0
        self.lambda_mask_entropy = 0.0
                
        self.lambda_ref_roughness_smooth = 0.01
        self.lambda_ref_strength_smooth = 0.01

        self.lambda_normal_render_depth = 0.02
        self.lambda_normal_smooth = 0.01
        self.lambda_white_light = 0.003

        self.lambda_rgb = 1

        self.lambda_dssim = 0.2
        self.lambda_pbr = 1
        self.lambda_light = 0.0
        self.lambda_base_color_smooth = 0.03
        self.lambda_roughness_smooth = 0.01
        self.lambda_metallic_smooth = 0.01
        self.lambda_env_smooth = 0.01
        self.lambda_reflect_strength_equal_metallic = 0.0

        self.use_env_scope = False
        self.env_scope_center = [0.,0.,0.]
        self.env_scope_radius = 0.0

        self.without_normal_propagation = False 

        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)
    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
