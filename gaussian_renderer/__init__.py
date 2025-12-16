from gaussian_renderer.render import render
from gaussian_renderer.render_fast import render_fast


render_fn_dict = {
    "render_ref": render,
    "render_ref_pbr": render,
    "render_ref_fast": render_fast
}