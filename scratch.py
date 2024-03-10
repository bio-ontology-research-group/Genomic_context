#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psutil

def get_free_memory():
    memory = psutil.virtual_memory()
    return memory.available / (1024.0 ** 3)  # Convert bytes to gigabytes

print(f"Free CPU Memory: {get_free_memory():.2f} GB")


# In[2]:


import torch
torch.backends.cuda.matmul.allow_tf32 = True
from datasets import Dataset
import os
from deepgo.utils import Ontology


# In[3]:


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()


# In[4]:


tokenizer_path = "WordLevel_tokenizer_trained_InterPro.json"
tokenizer = tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(512)


# In[5]:


tokenizer.get_vocab_size()


# In[6]:


test = tokenizer.encode("WP_265490204 WP_206642677 WP_053312998 WP_251959347 WP_000076573 WP_227526754 WP_218401808 WP_106925592")
test.ids


# In[7]:


import json
with open("/home/toibazd/Most_frequent_IPs.json", "r") as f:
    ips = json.load(f)

sorted_dict = sorted(ips.items(), key=lambda x: x[1], reverse=True)
most_frequent_ips = [item[0] for item in sorted_dict[100:300]]


# In[8]:


print(most_frequent_ips[:10])


# In[9]:


from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
import csv
from tqdm.auto import tqdm
import numpy as np



ip_to_go = defaultdict(list)
data_dict = defaultdict(list)
enc = MultiLabelBinarizer()
new_tsv_filename = "/home/toibazd/Family_IPs_with_GO.tsv"
go = Ontology('data/go.obo')


with open(new_tsv_filename, "r") as new_tsvfile:
    reader = csv.reader(new_tsvfile, delimiter="\t")
    next(reader)
    for row in tqdm(reader):
        ip = row[0]  # Assuming the IP is in the first column
        go_terms = row[6]  # Assuming the GO terms are in the second column

        # Add IP and corresponding GO terms to data_dict
        ip_to_go[ip]+= go_terms.split(',')


with open("/home/toibazd/Prot2IP_GO_filtered_MF.tsv", "r") as tsvfile:
    reader = csv.reader(tsvfile, delimiter = "\t")
    for row in tqdm(reader):
        key = row[0].split("prot_")[1].split(".")[0]
        iprs = eval(row[1])
        

        # Save only if there are filtered InterPro IDs
        for ip in iprs:
            if ip in most_frequent_ips and ip_to_go[ip]:
                for GO in ip_to_go[ip]:
                    data_dict[key].append(GO)
#                     data_dict[key].extend(list(go.get_ancestors(GO)))


# In[10]:


len(data_dict)


# In[11]:


all_values = [value for values in data_dict.values() for value in values]

# Convert the list into a set to remove duplicates
unique_go = set(all_values)

print("Number of unique words:", len(unique_go))


# In[12]:


from sklearn.preprocessing import MultiLabelBinarizer
enc = MultiLabelBinarizer()
enc.fit(ip_to_go.values())
one_hot_encoded = enc.transform(data_dict.values())
one_hot_encoded_dict = {key: value for key, value in zip(data_dict.keys(), one_hot_encoded)}

print(len(one_hot_encoded_dict.keys()))


# In[13]:


len(one_hot_encoded_dict)


# In[14]:


len(one_hot_encoded)


# In[15]:


# Find unique numbers and their counts
unique_numbers, counts = np.unique(one_hot_encoded, return_counts=True)
all_count = 0
# Print the count of each number
for number, count in zip(unique_numbers, counts):
    all_count+=count
    print(f"Number {number}: Count {count}")
print(all_count)


# In[16]:


import os
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

directory = '/ibex/user/toibazd/InterPro_annotated_genomes/'
one_hot_encoded_sentences = {}

sentence_length = 40
sentences_per_IP = 200

# Set random seed for reproducibility
random.seed(42)

# Randomly choose 1000 files with seed 42
selected_files = os.listdir(directory)


# Define a function to process a file
def process_file(filename, IP):
    sentences = []

    filepath = os.path.join(directory, filename)

    with open(filepath, 'r') as file:
        content = file.read()
        words = content.strip().split()

        # Check if the key is in the file
        for i in range(19, len(words)-20):
            # Shuffle the indices of the words containing the key
            if IP in data_dict[words[i]]:
                if len(words) - i >= 21:
                    sentence = " ".join(words[i - 19:i + sentence_length - 19])
                    sentences.append(sentence)
    return sentences


# Iterate over keys
for IP in tqdm(unique_go):
    one_hot_encoded_sentences[IP] = []
    sentences_count = 0

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_file, filename, IP) for filename in selected_files]
        for future in futures:
            sentences = future.result()
            one_hot_encoded_sentences[IP].extend(sentences)
            sentences_count += len(sentences)
            if sentences_count >= sentences_per_IP:
                break

    # Break if the required number of sentences per key is reached


# In[ ]:


import json

with open('BERT_DNN_senteces_with_GO.json', 'w') as f:
    json.dump(one_hot_encoded_sentences, f)
print("Done")


# In[ ]:




