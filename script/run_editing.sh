#!/bin/bash
root_dir=""
dataset_name="tensoIR"
list="armadillo hotdog lego ficus"
output_path=""

gpu_id="0"


for i in $list
do
    CUDA_VISIBLE_DEVICES=$gpu_id python render_and_eval.py \
        -m ${output_path}/${dataset_name}/${i}/stage2/ \
        -c ${output_path}/${dataset_name}/${i}/stage2/checkpoint/chkpnt40000.pth \
        --occlusion_path ${output_path}/${dataset_name}/${i}/stage1/checkpoint/occlusion_volumes.pth \
        --ref_map \
        --compute_with_prt \
        --skip_eval \
        -t render_ref_pbr \
        --metallic \
        --save_name editing_material \
        -w \
        --editing_config_path  ${output_path}/${dataset_name}/${i}/stage2/editing_config.json \
        --save_video

    CUDA_VISIBLE_DEVICES=$gpu_id python eval_relighting_tensorIR.py \
        -m ${output_path}/${dataset_name}/${i}/stage2 \
        -c ${output_path}/${dataset_name}/${i}/stage2/checkpoint/chkpnt40000.pth \
        --occlusion_path ${output_path}/${dataset_name}/${i}/stage1/checkpoint/occlusion_volumes.pth \
        --ref_map \
        --relight \
        --compute_with_prt \
        --metallic \
        -t render_ref_pbr \
        --skip_eval \
        --save_video \
        --save_name editing_material_relight \
        --no_rescale_albedo \
        -w \
        --editing_config_path  ${output_path}/${dataset_name}/${i}/stage2/editing_config.json
       
done
