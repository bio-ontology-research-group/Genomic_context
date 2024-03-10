#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J AA
#SBATCH -o AA.%J.out
#SBATCH -e AA.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=10:30:00
#SBATCH --mem=30G

#run the application:
source activate base
python scratch.py
