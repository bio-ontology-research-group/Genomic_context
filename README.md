<div align=center>
<img src="figures/logo.png" width="400px">
</div>

<h2 align="center"> <a href="https://www.biorxiv.org/content/10.1101/2024.10.14.618363v1">Context-based protein function prediction in bacterial genomes</a></h2>

<h5 align="center"><p>This project was previously named LLaVA-o1. We have renamed it to LLaVA-CoT. </h5>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.</h5>

<h5 align=center>

[![hf](https://img.shields.io/badge/🤗-Hugging%20Face-blue.svg)](https://huggingface.co/Dauka-transformers/interpro_bert_2)
[![arXiv](https://img.shields.io/badge/Arxiv-2411.10440v2-b31b1b.svg?logo=arXiv)](https://www.biorxiv.org/content/10.1101/2024.10.14.618363v1)
[![License](https://img.shields.io/badge/Code%20License-Apache2.0-yellow)](https://github.com/PKU-YuanGroup/LLaVA-CoT/blob/main/LICENSE)

</h5>


We explored a novel context-only dependent protein function prediction by leveraging the transformer-based representation learning on bacterial genomic context. This repository contains scripts which we used to train BERT model, along with scripts we used for function prediction and evaluation of contextual approach.

## Installation
* The code was developed and tested using Python 3.9.13
* Clone the repository
```terminal
git clone https://github.com/bio-ontology-research-group/Genomic_context.git
```
* Create conda environment
```terminal
conda create --name genomic_context python=3.9.13
```
* Activate your environment
```terminal
conda activate genomic_context
```
* Install dependencies
```terminal
pip install -r requirements.txt
```
* The training data we used in this study is deposited in Zenodo database under accession code 10.5281/zenodo.13932747 ([https://doi.org/10.5281/zenodo.13932747](https://doi.org/10.5281/zenodo.13932747)). Data includes NLP formatted genomes, cluster-representative protein sequences and mmseqs2 clustering results.


## Repo guide
- train - contains scripts to generate training data and train BERT. The training data is generated from [genomic corpus](https://doi.org/10.5281/zenodo.7047944).


- BERT_word2vec_benchmark - contains scripts to run BERT and word2vec evaluations. The genome corpus for evaluation can be obtained via following [link](https://doi.org/10.5281/zenodo.7047944). Pre-trained BERT model exported to [HF Hub](https://huggingface.co/Dauka-transformers/BERT_word2vec)
To get BERT classification results, run:
```terminal
python BERT_eval.py --word_to_label_mapping word_to_label.csv --directory Path_to_NLP_formatted_genomes
```
For word2vec classification results, run:
```terminal
python word2vec_eval.py --word_to_label_mapping word_to_label.csv --directory Path_to_NLP_formatted_genomes --word2vec_model Path_to_word2vec_model
```

- Defense_InterPro's - contains tsv files with InterPro ID's annotating bacterial defense systems. Data obtained from [InterPro website](https://www.ebi.ac.uk/interpro/entry/InterPro/#table)
- Secretion_InterPro's - contains tsv file with InterPro ID's annotating bacterial secretion systems. Data obtained from [InterPro website](https://www.ebi.ac.uk/interpro/entry/InterPro/#table)
- 
## Citations

If you find this work useful in your work, please cite our paper:
```bibtex
@article{Toibazar2024,
  title = {Context-based protein function prediction in bacterial genomes},
  url = {http://dx.doi.org/10.1101/2024.10.14.618363},
  DOI = {10.1101/2024.10.14.618363},
  publisher = {Cold Spring Harbor Laboratory},
  author = {Toibazar,  Daulet and Kulmanov,  Maxat and Hoehndorf,  Robert},
  year = {2024},
  month = oct 
}
```
