

# For spatial only
CUDA_VISIBLE_DEVICES=1 python code/train.py --setting spatial --batch_size 4 --epoch 600 --dst_list ToothFairy --num_views 10 --combine mlp --logdir ./logs --dst_root /cm/archive/cuongtv42/C2RV-CBCT/data -trunc LH-HH LH-HH LH-HH LH-HH -sobel 0 0 0 0 -patch 1 1 1 1 -psize 16 16 16 16 -fac dep-sep dep-sep dep-sep dep-sep -attn 1 1 1 1 -grid linear --use_wandb 


# For DuFal:
CUDA_VISIBLE_DEVICES=1 python code/train.py --setting rerun-LH_HH-0_sobel-1_patch-16_psize-dep_sep_fac-1_attn-linear_grid-8_fuse --batch_size 4 --epoch 600 --dst_list ToothFairy --num_views 10 --combine mlp --logdir ./logs --dst_root /cm/archive/cuongtv42/C2RV-CBCT/data -trunc LH-HH LH-HH LH-HH LH-HH -sobel 0 0 0 0 -patch 1 1 1 1 -psize 16 16 16 16 -fac dep-sep dep-sep dep-sep dep-sep -attn 1 1 1 1 -grid linear -fuse 8 -freq enc -fno --use_wandb 
CUDA_VISIBLE_DEVICES=1 python code/evaluate.py --epoch 540 --dst_list ToothFairy --dst_root /cm/archive/cuongtv42/C2RV-CBCT/data --split test --combine mlp --num_views 10 --visualize --logdir ./logs --setting rerun-LH_HH-0_sobel-1_patch-16_psize-dep_sep_fac-1_attn-linear_grid-8_fuse -trunc LH-HH LH-HH LH-HH LH-HH -sobel 0 0 0 0 -patch 1 1 1 1 -psize 16 16 16 16 -fac dep-sep dep-sep dep-sep dep-sep -attn 1 1 1 1 -grid linear -fuse 8 -fno -freq enc
