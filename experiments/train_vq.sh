#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/train_vq.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask
name='1_uplow_resetEvery_fullEmb_q1024'
python train_vq.py \
    --name ${name} \
    --gpu_id 0 \
    --gamma 0.05 \
    --lr 2e-4 \
    --dataset_name t2m \
    --batch_size 256 \
    --num_quantizers 6  \
    --max_epoch 50 \
    --quantize_dropout_prob 0.2

sleep 500
# --commit 0.01 \

# MoMask pretrain
# --gamma 0.05 \
# --lr 2e-4 \
# --num_quantizers 6 \
