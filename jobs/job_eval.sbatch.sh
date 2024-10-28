#!/bin/bash -l

#SBATCH -J edsl_eval
#SBATCH -o ../logs/log_eval.txt
#SBATCH -c 40
#SBATCH --exclusive

enable_lmod
module load container_env pytorch-gpu/1.9.0

crun -p ~/envs/fugep python ../scripts/script_eval.py
