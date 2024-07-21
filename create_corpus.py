#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
from tqdm import tqdm 
def create_sentences(directory):
    sentences = []
    
    # Iterate over the text files in the directory
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Read the contents of the file
            with open(file_path, 'r') as file:
                content = file.read()
                words = content.split()
                
                # Generate sentences of length 9 with step size 3
                for i in range(0, len(words) - 8, 3):
                    sentence = ' '.join(words[i:i+9])
                    sentences.append(sentence)
    
    # Shuffle the sentences randomly
    random.shuffle(sentences)
    
    # Split the sentences into train and test sets
    train_size = int(0.9 * len(sentences))
    train_sentences = sentences[:train_size]
    test_sentences = sentences[train_size:]
    
    # Randomly remove 10% of sentences from train set
    # train_remove_count = int(0.1 * len(train_sentences))
    # train_sentences = random.sample(train_sentences, len(train_sentences) - train_remove_count)
    
    # Randomly remove 10% of sentences from test set
    # test_remove_count = int(0.1 * len(test_sentences))
    # test_sentences = random.sample(test_sentences, len(test_sentences) - test_remove_count)
    
    # Write the train sentences to train.txt
    with open('train.txt', 'w') as train_file:
        train_file.write('\n'.join(train_sentences))
    
    # Write the test sentences to test.txt
    with open('test.txt', 'w') as test_file:
        test_file.write('\n'.join(test_sentences))
    
    print(f"Created {len(train_sentences)} sentences in train.txt")
    print(f"Created {len(test_sentences)} sentences in test.txt")

# Specify the directory containing the text files
directory = '/ibex/user/toibazd/InterPro_annotated_genomes/'

# Create the sentences and write them to files
create_sentences(directory)


# In[ ]:





# In[ ]:





# In[ ]:




