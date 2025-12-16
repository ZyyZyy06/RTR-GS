#!/bin/bash
root_dir=""
dataset_name="tensoIR"
list="hotdog ficus lego armadillo"
output_path="./lab_output/"

gpu_id="0"

for i in $list
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --eval \
        -s ${root_dir}${i} \
        -m ${output_path}/${dataset_name}/${i}/stage1 \
        --lambda_mask_entropy 0.1 \
        --diffuse_iteration 3000 \
        --ref_map \
        --skip_eval \
        -t neilf_ref \
        --compute_with_prt

    CUDA_VISIBLE_DEVICES=$gpu_id python baking.py \
        --checkpoint ${output_path}/${dataset_name}/${i}/stage1/checkpoint/chkpnt30000.pth \
        --bound 1.5 \
        --occlu_res 128

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --eval \
        -s ${root_dir}${i} \
        -m ${output_path}/${dataset_name}/${i}/stage2 \
        -c ${output_path}/${dataset_name}/${i}/stage1/checkpoint/chkpnt30000.pth \
        --occlusion_path ${output_path}/${dataset_name}/${i}/stage1/checkpoint/occlusion_volumes.pth \
        --iterations 40000 \
        --lambda_mask_entropy 0.1 \
        --lambda_reflect_strength_equal_metallic 0.1 \
        --skip_eval \
        --metallic \
        --ref_map \
        -t neilf_ref_pbr \
        --compute_with_prt

    CUDA_VISIBLE_DEVICES=$gpu_id python render_and_eval.py \
        -m ${output_path}/${dataset_name}/${i}/stage2 \
        -c ${output_path}/${dataset_name}/${i}/stage2/checkpoint/chkpnt40000.pth \
        --occlusion_path ${output_path}/${dataset_name}/${i}/stage1/checkpoint/occlusion_volumes.pth \
        --ref_map \
        --compute_with_prt \
        --metallic \
        -t neilf_ref_pbr \
        --save_video


    CUDA_VISIBLE_DEVICES=$gpu_id python eval_relighting_tensorIR.py \
        -m ${output_path}/${dataset_name}/${i}/stage2 \
        -c ${output_path}/${dataset_name}/${i}/stage2/checkpoint/chkpnt40000.pth \
        --occlusion_path ${output_path}/${dataset_name}/${i}/stage1/checkpoint/occlusion_volumes.pth \
        --ref_map \
        --relight \
        --compute_with_prt \
        --metallic \
        -t neilf_ref_pbr \
        --save_video    
done