import os
import sys
from tqdm import tqdm


gpuid = '0' # Default to GPU 0
task_name = f"gscream_v1" # Renamed task
refs_root = './refs'

voxel_size=0.005
update_init_factor=16
os.system("ulimit -n 4096")

# Only run for scene "1" as an example, or you can restore the full list
target_scenes = ["1"] 

for nn in target_scenes:

    # Find reference files
    ref_images = [i for i in sorted(os.listdir(refs_root)) if i.startswith(nn+'_out') and i.endswith('png')]
    ref_depths = [i for i in sorted(os.listdir(refs_root)) if i.startswith(nn+'_out') and i.endswith('npy')]
    
    if len(ref_images) == 0 or len(ref_depths) == 0:
        print(f"Skipping scene {nn}: Reference files not found in {refs_root}")
        continue

    ref_image_path = os.path.join(refs_root, ref_images[0])
    print(f"Ref Image: {ref_image_path}")
    assert os.path.exists(ref_image_path), ref_image_path
    
    ref_depth_path = os.path.join(refs_root, ref_depths[0])
    print(f"Ref Depth: {ref_depth_path}")
    assert os.path.exists(ref_depth_path), ref_depth_path

    image_root = f"images"  # by default

    # Added selfattn arguments
    cmd = f"CUDA_VISIBLE_DEVICES={gpuid} CUDA_LAUNCH_BLOCKING=1 python -u train.py \
        -s ./data/spinnerf_dataset_processed/{nn} \
        --eval \
        --lod 0 \
        --gpu -1 \
        --voxel_size {voxel_size} \
        --update_init_factor {update_init_factor} \
        --iterations 30_000 \
        --port 10001 \
        -m outputs/spinnerf_datasetv1/{nn}/{task_name}/ \
        --is_spin \
        --images {image_root} \
        --specified_ply_path ./data/spinnerf_dataset_processed/{nn}/sparse/0/points3D.ply \
        --ref_image_path {ref_image_path} \
        --ref_depth_path {ref_depth_path} \
        --load_mask \
        --load_depth \
        --load_norm \
        --lpips_lr 0 \
        --lpips_b 20 \
        --perceptual_lr 0 \
        --perceptual_b 2 \
        --refer_rgb_lr 1.0 \
        --refer_rgb_lr_fg 20.0 \
        --other_rgb_lr 1.0 \
        --other_rgb_lr_fg 0.0 \
        --refer_depth_lr 1 \
        --refer_depth_lr_fg 100 \
        --refer_depth_lr_smooth 1 \
        --other_depth_lr 0.1 \
        --other_depth_lr_smooth 0.1 \
        --refer_normal_lr 0 \
        --other_normal_lr 0 \
        --refer_opacity_lr 0 \
        --other_opacity_lr 0 \
        --uncertainty_lr 0 \
        --vgg_lr 0 \
        --discriminator_lr 0 \
        --adv_lr 0 \
        --crossattn_lr_init 0.002 \
        --crossattn_lr_final 0.00002 \
        --crossattn_lr_delay_mult 0.01 \
        --crossattn_lr_max_steps 30000 \
        --selfattn_lr_init 0.002 \
        --selfattn_lr_final 0.00002 \
        --selfattn_lr_delay_mult 0.01 \
        --selfattn_lr_max_steps 30000 \
        --enable_crossattn_refview 1.0 \
        --enable_crossattn_otherview 1.0 \
        --attn_head_num 8 \
        --attn_head_dim 64 \
        --enable_enlarge_samping 0.0 \
        --sampling_2D_enlarge_ratio 2.0 \
        --enable_edge_samping 1.0 \
        --sampling_2D_small_ratio 0.6 \
        --enable_pe 0.0 \
        --crossattn_feat_update_ema 0.03 \
        --start_crossattn_from 15000 \
        > log_training/log_train_spin-{nn}_{task_name}.txt"
    
    print(f"Running command for scene {nn}...")
    os.system(cmd)
    # print(cmd)
