#!/bin/bash
#SBATCH --job-name="Example job name!"
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:A40:4
#SBATCH --partition=gpu
#SBATCH --mem=128G

echo "This job has started!"


module load miniforge3

mamba activate mm_sparse4d

module load gcc/10.2.0 
module load cuda/11.6


cd team_code_transfuser
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=2 --max_restarts=0 --rdzv_id=1234576890 --rdzv_backend=c10d train.py --logdir ./logdir --root_dir ./root_dir --parallel_training 1
echo "This job has ended! Yippy"