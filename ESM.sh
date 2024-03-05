#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J ESM
#SBATCH -o ESM.%J.out
#SBATCH -e ESM.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --constraint=v100
#SBATCH --mem-per-cpu=20GB
echo $OMP_NUM_THREADS

#run the application:
source activate base
module load cuda/11.7.1
cd /home/toibazd/Data/BERT/
python  ESM/extract.py esm2_t36_3B_UR50D ESM/few_proteins.fasta ESM/some_proteins_emb_esm2 --repr_layers 36 --include mean



