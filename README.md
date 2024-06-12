# PIVOT

# DynamicViT Baseline Running Script

This script is designed to run a baseline experiment using the DynamicViT model. Below are the variables used in the script and their explanations:

- `effort1`: A string defining effort levels for a specific task.
- `effort2`: A variable representing a default effort level.
- `effort3`: A variable representing a high effort level.
- `model`: The name of the model being used, here it's "lvvit-s".
- `batch_size`: The number of samples processed together in one iteration.
- `base_rate`: A string defining the baseline rate.
- `checkpoint1`: The path to the checkpoint file for the model trained with the specified effort level (`effort1`).
- `checkpoint2`: The path to the checkpoint file for the model trained with the default effort level (`effort2`).
- `checkpoint3`: The path to the checkpoint file for the model trained with the high effort level (`effort3`).

### Running the Script

To run the script, execute the following command:

```bash
python weighted_effort_infer.py --sharing1 $effort1 --sharing2 $effort2 --sharing3 $effort3 --keep_ratio 1_1_1 --batch_size $batch_size --model $model --effort1 $checkpoint1 --effort2 $checkpoint2 --effort3 $checkpoint3
