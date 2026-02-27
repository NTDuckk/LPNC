DATASET_NAME="RSTPReid"

# Single GPU training
# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
# --name LPNC \
# --output_dir 'LPNC_log' \
# --dataset_name $DATASET_NAME \
# --loss_names 'supid+cotrl+cid' \
# --num_epoch 60

# Multi-GPU training (using torchrun)
# Usage: ./run.sh <num_gpus>
# Example: ./run.sh 2  (for 2 GPUs)

NUM_GPUS=${1:-1}  # Default to 1 GPU if not specified

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=$NUM_GPUS \
train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--num_epoch 60
