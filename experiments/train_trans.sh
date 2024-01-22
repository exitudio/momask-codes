#!/bin/sh

# cd /home/epinyoan/git/momask-codes
# screen -S temp /home/epinyoan/git/momask-codes/experiments/train_trans.sh

. ~/miniconda3/etc/profile.d/conda.sh
cd /home/epinyoan/git/momask-codes
conda activate momask
name='3_GPT_randId0-.5_weightBySample_b512_mile50k80k_ep2k'
python train_t2m_transformer.py \
    --name ${name} \
    --gpu_id 6 \
    --dataset_name t2m \
    --milestones 50000 80000 \
    --batch_size 512 \
    --max_epoch 2000 \
    --vq_name rvq_nq6_dc512_nc512_noshare_qdp0.2



sleep 500
