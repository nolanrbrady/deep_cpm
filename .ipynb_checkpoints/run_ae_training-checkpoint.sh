#!/bin/bash
#SBATCH --job-name=train_rise_ae
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nolan.brady@colorado.edu
#SBATCH --output=./ae_training.out
#SBATCH --error=./ae_training.err
#SBATCH --time=04:00:00

# Testing running the script in parrallel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12

python ae_training.py
