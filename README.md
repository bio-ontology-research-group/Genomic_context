## Context-aware protein function prediction in bacterial genomes

We developed a novel context-only dependent protein function prediction method by leveraging the transformer model on bacterial genomic context. This repository contains scripts which we used to train BERT model, along with scripts we used for function prediction and evaluation of contextual approach.

## Dependencies
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
## Repo guide
- BERT_word2vec_benchmark - contains scripts to run BERT and word2vec evaluations. The corpus for evaluation can be obtained via this [link]{(https://doi.org/10.5281/zenodo.7047944}.


## Citations

If you find this work useful in your work, please cite our paper:
```bibtex

```
