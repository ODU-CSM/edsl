#!/bin/bash -l

#SBATCH -J edsl_train
#SBATCH -o ../logs/log_train.txt
#SBATCH -c 8
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH -C v100

export CUBLAS_WORKSPACE_CONFIG=":4096:8"


enable_lmod
module load container_env pytorch-gpu/1.9.0

crun -p ~/envs/fugep python ../scripts/script_train.py