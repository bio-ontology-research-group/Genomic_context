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
* The training data we used in this study is deposited in Zenodo database under accession code 10.5281/zenodo.13932747 ([https://doi.org/10.5281/zenodo.13932747](https://doi.org/10.5281/zenodo.13932747)). Data includes NLP formatted genomes, cluster-representative protein sequences and mmseqs2 clustering results.


## Repo guide
- BERT_word2vec_benchmark - contains scripts to run BERT and word2vec evaluations. The genome corpus for evaluation can be obtained via following [link](https://doi.org/10.5281/zenodo.7047944). Pre-trained BERT model exported to [HF Hub](https://huggingface.co/Dauka-transformers/BERT_word2vec)
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
