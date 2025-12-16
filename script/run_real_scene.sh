# for Mipnerf360 or shiny blender real
root_dir=""
dataset_name="real_scene"
list="garden bicycle stump room kitchen bonsai counter"
# list="gardenspheres sedan toycar"
output_path=""
gpu_id="0"

for i in $list
do
    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --eval \
        -s ${root_dir}${i} \
        --images ${root_dir}${i}/images_4 \
        -m ${output_path}/${dataset_name}/${i}/stage1 \
        --lambda_normal_render_depth 0.01 \
        --diffuse_iteration 3000 \
        --skip_eval \
        --ref_map \
        -t neilf_ref \
        --compute_with_prt \
        --densify_grad_threshold 0.0005

    CUDA_VISIBLE_DEVICES=$gpu_id python baking.py \
        --checkpoint ${output_path}/${dataset_name}/${i}/stage1/checkpoint/chkpnt30000.pth \
        --bound 2.0 \
        --occlu_res 128   

    CUDA_VISIBLE_DEVICES=$gpu_id python train.py --eval \
        -s ${root_dir}${i} \
        --images ${root_dir}${i}/images_4 \
        -m ${output_path}/${dataset_name}/${i}/stage2 \
        -c ${output_path}/${dataset_name}/${i}/stage1/checkpoint/chkpnt30000.pth \
        --occlusion_path ${output_path}/${dataset_name}/${i}/stage1/checkpoint/occlusion_volumes.pth \
        --save_training_vis \
        --iterations 40000 \
        --lambda_ref_strength_smooth 0.01 \
        --lambda_reflect_strength_equal_metallic 0.0 \
        --ref_map \
        -t neilf_ref_pbr \
        --compute_with_prt


    CUDA_VISIBLE_DEVICES=$gpu_id python eval_relighting_colmap.py --eval \
        -s ${root_dir}${i} \
        --images ${root_dir}${i}/images_4 \
        -m ${output_path}/${dataset_name}/${i}/stage2 \
        -c ${output_path}/${dataset_name}/${i}/stage2/checkpoint/chkpnt40000.pth \
        -e /data/zhouyongyang/dataset/tensorIR/env_maps/high_res_envmaps_1k/ \
        --ref_map \
        -t neilf_ref_pbr \
        --compute_with_prt \
        --save_video
done
