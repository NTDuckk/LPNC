DATASET_NAME="RSTPReid"

CUDA_VISIBLE_DEVICES=0 \
python train.py \
--name LPNC \
--output_dir 'LPNC_log' \
--dataset_name $DATASET_NAME \
--loss_names 'supid+cotrl+cid' \
--batch_size 64 \
--gradient_accumulation_steps 8 \
--num_epoch 60
