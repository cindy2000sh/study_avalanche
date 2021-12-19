#!/bin/sh
CUDA_VISIBLE_DEVICES=0,1 python main.py \
--root_dir "/data3/zhiqiul/clear_datasets/CLEAR10-TEST" \
--save_path "/data3/siqiz/avalanche/outputs" \
--debug "True" \
--num_buckets 11 \
--is_pretrained "False" \
--pretrained_path "/data/jiashi/moco_resnet50_clear_10_feature" \
--num_classes 11 \
--buffer_size "8,8" \
--train_mb_size 10 \
--train_epochs 10 \
--eval_mb_size 10 \
--lr 0.001 \
--strategies "AGEM" \
--alpha 1.0 \
--biased_mode "dynamic"