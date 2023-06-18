#!/usr/bin/env bash

DATASET="../mvs_training/dtu"

LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" 

OUT_DIR="./outputs"

python main.py --extract_geometry \
--test_n_view 3 --test_ray_num 400 --volume_reso 96 \
--root_dir=$DATASET --load_ckpt=$LOAD_CKPT --out_dir=$OUT_DIR $@
