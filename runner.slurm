#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J get_classifier                              # job name
#SBATCH --chdir=/home/esragenc/get-classification     # working directory
#SBATCH --gres=gpu:2                                  # request 2 GPUs
#SBATCH --output=/home/esragenc/get_classifier-%j.out # output file
#SBATCH --error=/home/esragenc/get_classifier-%j.err  # error file
#SBATCH --time=25:00:00                              # max job time

# Activate conda environment
source /home/esragenc/anaconda3/bin/activate GET

# Get the master node's hostname
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export WORLD_SIZE=2
export CUDA_VISIBLE_DEVICES=0,1

# Print debug information
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the training script with torchrun
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_classifier.py \
    --batch_size 128 \
    --epochs 200 \
    --lr 0.001 \
    --hidden_size 384 \
    --output_dir checkpoints/slurm_run

echo "Training completed" 