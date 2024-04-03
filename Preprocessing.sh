#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J Preprocessing
#SBATCH -o Preprocessing.%J.out
#SBATCH -e Preprocessing.%J.err
#SBATCH --mail-user=daulet.toibazar@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:30:00
#SBATCH --mem=250G
#SBATCH --constraint=local_200G

#run the application:
source activate base
python BERT_data_preprocessing.py
