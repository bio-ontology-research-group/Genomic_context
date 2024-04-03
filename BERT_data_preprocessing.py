#!/usr/bin/env python
# coding: utf-8

# In[4]:


import psutil

def get_free_memory():
    memory = psutil.virtual_memory()
    return memory.available / (1024.0 ** 3)  # Convert bytes to gigabytes

print(f"Free CPU Memory: {get_free_memory():.2f} GB")


# In[5]:


import torch
torch.backends.cuda.matmul.allow_tf32 = True
from datasets import Dataset
import os


# In[6]:


def divide_into_sentences(input_string, sentence_length=9, step_size=5):
    words = input_string.split()
    if len(words) < sentence_length:
        return [" ".join(words)]

    sentences = []
    for i in range(0, len(words) - sentence_length + 1, step_size):
        sentences.append(" ".join(words[i: i + sentence_length])) 

    return sentences



def split_data(data, train_ratio=0.7, validate_ratio=0.1, test_ratio=0.2, seed=123):
    random.seed(seed)
    random.shuffle(data)
    n = len(data)
    train_end = int(train_ratio * n)
    validate_end = int((train_ratio + validate_ratio) * n)
    train_data = data[:train_end]
    validate_data = data[train_end:validate_end]
    test_data = data[validate_end:]
    return train_data, validate_data, test_data



# In[7]:


from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer.pre_tokenizer = WhitespaceSplit()


# In[8]:


tokenizer_path = "WordLevel_tokenizer_trained_InterPro.json"
tokenizer = tokenizer.from_file(tokenizer_path)
tokenizer.enable_truncation(256)


# In[9]:


tokenizer.get_vocab_size()


# In[10]:


test = tokenizer.encode("WP_265490204 WP_206642677 WP_053312998 WP_251959347 WP_000076573 WP_227526754 WP_218401808 WP_106925592")
test.ids


# In[11]:


import random
from tqdm import tqdm
import re


all_sentences = []
absolute_path_genomes = "/ibex/user/toibazd/InterPro_annotated_genomes/"

for file_name in tqdm(os.listdir(absolute_path_genomes)):
    short_sentences = []
    with open(os.path.join(absolute_path_genomes, file_name), "r", encoding="latin-1") as infile:
        content = infile.read()
        short_sentences = divide_into_sentences(content)
        all_sentences+=short_sentences


train_sentences, validation_sentences, test_sentences = split_data(all_sentences)

print(f"Free CPU Memory: {get_free_memory():.2f} GB")
print(len(train_sentences))


# In[12]:


train_inputs = tokenizer.encode_batch(train_sentences)
val_inputs = tokenizer.encode_batch(validation_sentences)
print("Training sentences: ",len(train_inputs))
print("Evaluation sentences: ", len(val_inputs))

print(f"Free CPU Memory: {get_free_memory():.2f} GB")
del train_sentences
del validation_sentences
print(f"Free CPU Memory: {get_free_memory():.2f} GB")


# In[13]:


cpi = 0
for inp in tqdm(train_inputs):
    inp.pad(11,direction = 'right',pad_id = 3, pad_token = '[PAD]' )
    if len(inp.ids) != 11:
        cpi+=1
        print(len(inp.ids))
print("Checkpoint one")
print("Checkpoint count: ", cpi)
print(f"Free CPU Memory: {get_free_memory():.2f} GB")
cpi = 0
for inp in tqdm(val_inputs):
    inp.pad(11,direction = 'right',pad_id = 3, pad_token = '[PAD]' )
    if len(inp.ids) != 11 and len(inp.attention_mask) != 11:
        cpi+=1
        print(len(inp.ids))
print("Checkpoint one")
print("Checkpoint count: ", cpi)
print(f"Free CPU Memory: {get_free_memory():.2f} GB")


# In[14]:


train_input_ids = torch.tensor([i.ids for i in tqdm(train_inputs)])
val_input_ids = torch.tensor([i.ids for i in tqdm(val_inputs)])
print("Checkpoint two")
print(f"Free CPU Memory: {get_free_memory():.2f} GB")
train_attention_mask = torch.tensor([i.attention_mask for i in tqdm(train_inputs)])
val_attention_mask = torch.tensor([i.attention_mask for i in tqdm(val_inputs)])
print("Checkpoint three")
print(f"Free CPU Memory: {get_free_memory():.2f} GB")


# In[15]:


train_ins = {"input_ids":train_input_ids,'attention_mask':train_attention_mask}
val_ins = {"input_ids":val_input_ids,'attention_mask':val_attention_mask}


print("Making MASK for Training")
train_ins["labels"] = train_ins["input_ids"].detach().clone()
# train_ins['labels'] = torch.zeros_like(train_ins['input_ids'])
rand = torch.rand(train_ins['input_ids'].shape)
train_mask_arr = (rand<0.18)*(train_ins['input_ids'] != 1) * (train_ins['input_ids'] != 3) *(train_ins['input_ids'] != 2)

print("Making MASK for Validation")
val_ins["labels"] = val_ins["input_ids"].detach().clone()
rand = torch.rand(val_ins['input_ids'].shape)
val_mask_arr = (rand<0.18)*(val_ins['input_ids'] != 1) * (val_ins['input_ids'] != 3) *(val_ins['input_ids'] != 2)


print(f"Free CPU Memory: {get_free_memory():.2f} GB")
del train_inputs
del val_inputs
print(f"Free CPU Memory: {get_free_memory():.2f} GB")


# In[16]:


selection = []

for i in tqdm(range(train_mask_arr.shape[0])):
    selection.append(
    torch.flatten(train_mask_arr[i].nonzero()).tolist()
    )
for i in range(train_mask_arr.shape[0]):
#     train_ins["labels"][i, selection[i]] = train_ins['input_ids'][i, selection[i]]
    train_ins['input_ids'][i, selection[i]]= 4
    
    
selection = []
print(f"Free CPU Memory: {get_free_memory():.2f} GB")
for i in tqdm(range(val_mask_arr.shape[0])):
    selection.append(
    torch.flatten(val_mask_arr[i].nonzero()).tolist()
    )
for i in range(val_mask_arr.shape[0]):
    val_ins['input_ids'][i, selection[i]]= 4
print(f"Free CPU Memory: {get_free_memory():.2f} GB")   


train_masked_indices = train_ins["input_ids"] != 4
train_ins["labels"][train_masked_indices] = -100

val_masked_indices = val_ins["input_ids"] != 4
val_ins["labels"][val_masked_indices] = -100



# In[17]:


import datasets
print(f"Free CPU Memory: {get_free_memory():.2f} GB")
train_dataset = datasets.Dataset.from_dict(train_ins)
val_dataset = Dataset.from_dict(val_ins)
print(f"Free CPU Memory: {get_free_memory():.2f} GB")
train_dataset.set_format("pt")
val_dataset.set_format("pt")


# In[18]:


print("Checkpoint one")


# In[19]:


train_dataset.save_to_disk("BERT_train_dataset_context5_no_pad")


# In[20]:


val_dataset.save_to_disk("BERT_val_dataset_context5_no_pad")


# In[21]:


print("Done")


# In[ ]:




