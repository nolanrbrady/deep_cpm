#!/bin/bash
#SBATCH --job-name=ae_dense  # Job name
#SBATCH --partition=gpu-a100-80g      # Partition
#SBATCH --nodes=1                     # Run all processes on a single node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --time=1-00:00:00               # Time limit hrs:min:sec
#SBATCH --output=tensorflow_%j.log   # Standard output and error log
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=nobr3541@colorado.edu     # Where to send mail
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=300G

 
# Load modules below for gpu such as CUDA
module load cuda11.8/toolkit/11.8.0
#module load tensorflow/2.15.0.post1  # Adjust this to the correct module name

# activate environment
#conda activate rise

srun python ae_training_dense.py
