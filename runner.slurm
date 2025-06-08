#!/bin/bash
#SBATCH --nodes=1
#SBATCH -J get_classifier_cifar100             # job name
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
    --data_path ./data \
    --name cifar100_get_classifier \
    --results_dir results_classifier \
    --model GET-Classifier-T \
    --input_size 32 \
    --num_classes 100 \
    --lr 1e-3 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --epochs 200 \
    --global_batch_size 256 \
    --eval_batch_size 256 \
    --global_seed 42 \
    --num_workers 2 \
    --mem \
    --log_every 100 \
    --ckpt_every 5000 \
    --eval_every 1000 \
    --f_solver anderson        \
    --f_max_iter 25 \
    --b_max_iter 25 \
    --f_tol 1e-3 \
    --b_tol 1e-3

echo "Training completed at $(date)" 