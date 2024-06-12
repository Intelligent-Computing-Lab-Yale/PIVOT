#!/bin/bash

epochs1=35
epochs2=10

base_rate=1_1_1
actual_share=6_9_12_15 
output_dir="/Your Output Path Goes Here"

python train.py --output_dir $output_dir --model lvvit-s --num_workers 4 --input_size 224 --batch_size 256 --epochs $epochs1 --keep_ratio $base_rate --warmup_epochs 5 --ratio_weight 2.0 --distill_weight 0.5 --clf_weight 1.0 --num_heads 6 --lr 0.01 --sharing $actual_share #--auto_resume True
