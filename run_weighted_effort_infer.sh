## Running Baseline


effort1=1_3_5_7_9_11_13_15
effort2=0
effort3=100

model=lvvit-s
batch_size=256


base_rate=1_1_1
checkpoint1="/home/am3554/project/DynamicViT/logs/ip_specific/lvvit/ip_specific_base_rate_${base_rate}_share_${effort1}/checkpoint-34.pth"
checkpoint2="/home/am3554/project/DynamicViT/logs/ip_specific/lvvit/ip_specific_base_rate_${base_rate}_share_${effort2}/checkpoint-0.pth"
checkpoint3="/home/am3554/project/DynamicViT/logs/ip_specific/lvvit/ip_specific_base_rate_${base_rate}_share_${effort3}/checkpoint-best.pth"

#checkpoint_deit_s="../pretrained/deit_small_patch16_224-cd65a155.pth"
python weighted_effort_infer.py --sharing1 $effort1 --sharing2 $effort2 --sharing3 $effort3 --keep_ratio 1_1_1 --batch_size $batch_size --model $model --effort1 $checkpoint1 --effort2 $checkpoint2 --effort3 $checkpoint3

#"../pretrained/deit_small_patch16_224-cd65a155.pth" #