#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/train_vq.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask
name='lfq_cdim18_ent.05_div1_cmmt1_6e-5_invT50_50ep_mileston150k250k_b256'
python train_vq.py \
    --name ${name} \
    --gpu_id 4 \
    --lr 6e-5 \
    --dataset_name t2m \
    --batch_size 256 \
    --num_quantizers 2  \
    --max_epoch 50 \
    --quantize_dropout_prob 0.2

sleep 500