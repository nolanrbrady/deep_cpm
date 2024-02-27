#!/bin/bash
#SBATCH --job-name=train_rise_ae
#SBATCH --partition=defq
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nolan.brady@colorado.edu
#SBATCH --output=./ae_training.out
#SBATCH --error=./ae_training.err
#SBATCH --time=048:00:00
#SBATCH --nodes=2
#SBATCH --mem=64G

ulimit -v 67108864


python ae_training.py
