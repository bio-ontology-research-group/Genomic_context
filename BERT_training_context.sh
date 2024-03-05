#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J BERT_F
#SBATCH -o BERT_InterPro_F.%J.out
#SBATCH -e BERT_InterPro_F.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=120:00:00
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=2
#SBATCH --constraint=v100
#SBATCH --mem-per-cpu=20GB
echo $OMP_NUM_THREADS

#run the application:
source activate base
module load cuda/11.7.1
cd /home/toibazd/Data/BERT/
accelerate launch --mixed_precision fp16 BERT_training-Copy1.py



