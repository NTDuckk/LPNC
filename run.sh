DATASET_NAME="RSTPReid"

# Single GPU training
# python train.py \
# --name LPNC \
# --output_dir 'LPNC_log' \
# --dataset_name $DATASET_NAME \
# --loss_names 'supid+cotrl+cid' \
# --num_epoch 60

# Multi-GPU training (using torchrun)
# Usage: ./run.sh <num_gpus>
# Example: ./run.sh 2  (for 2 GPUs)

NUM_GPUS=${1:-2}  # Default to 2 GPUs if not specified

torchrun --nproc_per_node=$NUM_GPUS \
train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--num_epoch 60
