#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J get_classifier                              # job name
#SBATCH --chdir=/home/esragenc/get-classification              # working directory
#SBATCH --gres=gpu:2                                  # request 2 GPUs
#SBATCH --output=/home/esragenc/get_classifier-%j.out # output file
#SBATCH --error=/home/esragenc/get_classifier-%j.err  # error file
#SBATCH --time=25:00:00                              # max job time

# Activate conda environment
source /home/esragenc/anaconda3/bin/activate GET

# Set distributed training parameters
export MASTER_PORT=12345
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

# Run the training script with torchrun
torchrun \
    --nproc_per_node=2 \
    --master_port=${MASTER_PORT} \
    train_classifier.py \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.001 \
    --hidden_size 384 \
    --output_dir checkpoints/slurm_run

echo "Training completed" 