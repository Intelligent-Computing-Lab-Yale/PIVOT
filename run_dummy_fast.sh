#!/bin/bash

# Define the function
replace_chars() {
  # Get the input arguments
  local original_string="-X-X-X-X-X-X-X-X-X-X-X-X"
  local replacements=("${@:1}")

  # Loop through the replacements
  for i in "${!replacements[@]}"; do
    # Get the current replacement
    local replacement="${replacements[i]}"
    words=(${replacement})
#    echo "$replacement"
    # Get the index to replace
    local index="${words[0]}"

    # Get the replacement character(s)
    local chars="${words[1]}"
#    echo $index
    # Replace the character(s) at the specified index
    original_string="${original_string:0:$((2*index+1))}${chars}${original_string:(2*$index+2)}"
  done

  # Return the updated string
  original_string="${original_string:0:0}""${original_string:$((1))}"
  echo "$original_string"
}

function convert_array_to_string() {
  local array=("$@") # Get all arguments as an array
  local string=$(printf "%s " "${array[@]}") # Convert array to space-separated string
  echo "$string"
}

# Example usage

function replace_spaces_with_underscores() {
  local string="$1"
  local modified_string="${string// /_}"
  echo "$modified_string"
}

epochs1=35
epochs2=10
echo "############## Running Dummy Fast ####################"


share_layers=6_9_12_15 #1_3_5_7_9_11
#share_layers=("3 A" "4 A" "5 A")
#share_layers_string=$(convert_array_to_string "${share_layers[@]}")
#directory_name=$(replace_spaces_with_underscores "$share_layers_string")
#sharing=$(replace_chars "${share_layers[@]}")
base_rate=1_1_1
actual_share=6_9_12_15 #4_6_7_8_9_10_11_12_13_15
output_dir="/home/am3554/project/DynamicViT/logs/ip_specific/ip_specific_base_rate_${base_rate}_share_${actual_share}"
checkpoint="/home/am3554/project/DynamicViT/logs/ip_specific/ip_specific_base_rate_${base_rate}_share_${share_layers}/checkpoint-33.pth"

python train.py --output_dir $output_dir --model lvvit-s --num_workers 4 --input_size 224 --batch_size 256 --epochs $epochs1 --keep_ratio $base_rate --warmup_epochs 5 --ratio_weight 2.0 --distill_weight 0.5 --clf_weight 1.0 --num_heads 6 --lr 0.01 --sharing $actual_share #--auto_resume True #--resume $checkpoint