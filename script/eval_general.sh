#!/usr/bin/env bash


DATASET="../Rectified_colmap"

LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" 

OUT_DIR="./outputs_g_final"

python main.py --extract_geometry --test_general --n_randomly_generated_views 6 --test_n_view 5 --test_ray_num 400 --volume_reso 96 --test_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
