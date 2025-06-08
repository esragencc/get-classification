#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J get_overfit_one_batch                # job name
#SBATCH --chdir=/home/esragenc/get-classification # working directory
#SBATCH --gres=gpu:1                              # request 1 GPU
#SBATCH --output=/home/esragenc/get_overfit-%j.out # output file
#SBATCH --error=/home/esragenc/get_overfit-%j.err  # error file
#SBATCH --time=01:00:00                            # max job time

# Activate conda environment
source /home/esragenc/anaconda3/bin/activate GET

# Print debug information
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the overfit test script
python overfit_one_batch.py

echo "Overfit one batch test completed at $(date)" 