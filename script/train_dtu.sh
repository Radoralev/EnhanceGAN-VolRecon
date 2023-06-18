#!/usr/bin/env bash

DATASET="../mvs_training/dtu/"

LOG_DIR="./checkpoints"
LOAD_CKPT="checkpoints/epoch=15-step=193199.ckpt" 
OUT_DIR="./outputs"

python main.py --max_epochs 1 --batch_size 2 --lr 0.0001 \
--weight_rgb 1.0 --weight_depth 1.0 \
--train_ray_num 512 --volume_reso 96 \
--root_dir=$DATASET --logdir=$LOG_DIR $@
