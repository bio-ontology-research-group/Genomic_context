#!/usr/bin/env python
# coding: utf-8

import csv
from tqdm import tqdm
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import multiprocessing
import json
from sklearn.preprocessing import MultiLabelBinarizer
import click

@click.command()
@click.option('--input_file', default="resluts_blastp.m8", help="Blastp results between experimental annotation proteins and study corpus proteins.", required=True, type=click.Path(exists=True))
@click.option('--ont', type=click.Choice(['cc', 'bp', 'mf']), required=True, help="Ontology type: cc (Cellular Component), bp (Biological Process), or mf (Molecular Function)")
@click.option('--goa_file', required=True, help="tsv file with Gene Ontology annotations", type=click.Path(exists=True))
@click.option('--directory', help="Directory to genome corpus", type=click.Path(exists=True))
def main(input_file, ont, goa_file, directory):
    # Existing functions remain the same
    def process_m8_file(input_file):
        mapping = {}
        with open(input_file, 'r') as infile:
            reader = csv.reader(infile, delimiter='\t')
            for row in tqdm(reader):
                if row[1] not in mapping:    
                    mapping[row[1]] = row[0]
        return mapping

    def process_goa_file(protein_dict, goa_file):
        result_dict = {}
        with open(goa_file, 'r') as f:
            tsv_reader = csv.reader(f, delimiter='\t')
            for row in tsv_reader:
                if len(row) >= 5:
                    protein_name = row[1]
                    if protein_name in protein_dict:
                        protein_symbol = protein_dict[protein_name]
                        go_term = row[4]
                        if protein_symbol not in result_dict:
                            result_dict[protein_symbol] = set()
                        result_dict[protein_symbol].add(go_term)
        return result_dict

    def process_file(filename, proteins_set, sentence_length, sentences_per_go, directory):
        sentences_dict = defaultdict(list)
        filepath = os.path.join(directory, filename)
        with open(filepath, 'r') as file:
            content = file.read()
            words = content.strip().split()
            for i in range(4, len(words) - 4):
                if words[i] in proteins_set:
                    sentence = " ".join(words[i - 4:i + sentence_length - 4])
                    assert len(sentence.split()) == 9, "The length of the split sentence is not equal to 9."
                    sentences_dict[words[i]].append(sentence)
        return sentences_dict

    def merge_dicts(dict1, dict2):
        for key, value in dict2.items():
            dict1[key].extend(value)
            if len(dict1[key]) > sentences_per_go:
                dict1[key] = dict1[key][:sentences_per_go]

    # Main execution
    filtered_proteins = process_m8_file(input_file)
    results = process_goa_file(filtered_proteins, goa_file)

    enc = MultiLabelBinarizer()
    one_hot_encoded = enc.fit_transform(results.values())
    one_hot_encoded_dict = {key: value for key, value in zip(results.keys(), one_hot_encoded)}

    print("Number of proteins that are one_hot_encoded", len(one_hot_encoded_dict.keys()))

    train_proteins = list(results.keys())
    test_proteins = list(results.keys())

    sentence_length = 9
    sentences_per_go = 400

    random.seed(42)

    selected_files = os.listdir(directory)
    random.shuffle(selected_files)

    one_hot_encoded_sentences = defaultdict(list)
    proteins_set = set(train_proteins)
    batch_size = 1000
    num_batches = len(selected_files) // batch_size + (1 if len(selected_files) % batch_size != 0 else 0)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for batch_idx in tqdm(range(num_batches)):
            batch_files = selected_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            futures = [executor.submit(process_file, filename, proteins_set, sentence_length, sentences_per_go, directory) for filename in batch_files]

            for future in as_completed(futures):
                sentences_dict = future.result()
                merge_dicts(one_hot_encoded_sentences, sentences_dict)

    for key in one_hot_encoded_sentences:
        if len(one_hot_encoded_sentences[key]) > sentences_per_go:
            one_hot_encoded_sentences[key] = one_hot_encoded_sentences[key][:sentences_per_go]

    one_hot_encoded_sentences = dict(one_hot_encoded_sentences)

    with open(f'BERT_DNN_senteces_with_GO_{ont}.json', 'w') as f:
        json.dump(one_hot_encoded_sentences, f)

    # Processing for test data
    one_hot_encoded_sentences_2 = defaultdict(list)
    sentence_per_IP = 1
    proteins_set = set(test_proteins)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        for batch_idx in tqdm(range(num_batches)):
            batch_files = selected_files[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            futures = [executor.submit(process_file, filename, proteins_set, sentence_length, sentence_per_IP, directory) for filename in batch_files]

            for future in as_completed(futures):
                sentences_dict = future.result()
                merge_dicts(one_hot_encoded_sentences_2, sentences_dict)

    one_hot_encoded_sentences_2 = dict(one_hot_encoded_sentences_2)

    with open(f'BERT_DNN_senteces_with_GO_{ont}_test.json', 'w') as f:
        json.dump(one_hot_encoded_sentences_2, f)

if __name__ == "__main__":
    main()