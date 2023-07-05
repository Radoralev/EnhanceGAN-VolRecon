#!/bin/bash

# Check if input arguments are provided
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 start end"
    exit 1
fi

# Input arguments
start=$1
end=$2

# Loop from start to end
for i in $(seq $start $end); do
    python evaluate_cgan.py --ckpt_netD checkpoints_cgan/checkpoint_${i}_netD.pth --ckpt_netG checkpoints_cgan/checkpoint_${i}_netG.pth --out metrics_${i}.csv
done
