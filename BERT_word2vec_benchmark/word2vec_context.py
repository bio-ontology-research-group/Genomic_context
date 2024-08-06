#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import click


@click.command()
@click.option('--word_to_label_mapping', type=click.Path(exists=True), default = "word_to_label.csv", help = "Word to label mapping csv file")
@click.option('--directory', type=click.Path(exists=True), help = "NLP formatted genome files directory (Expected to be in txt format)")
@click.option('--word2vec_model', type=click.Path(exists=True), help='Path to the pre-trained word2vec model', default = "gene2vec_w5_v300_tf24_annotation_extended_2021-10-03.w2v")
def main(word_to_label_mapping, directory, word2vec_model):
    df = pd.read_csv(word_to_label_mapping)
    
    # Specify the protein classes
    protein_classes = ['Amino sugar and nucleotide sugar metabolism', 'Benzoate degradation', 'Energy metabolism', 'Oxidative phosphorylation', 'Porphyrin and chlorophyll metabolism', 'Prokaryotic defense system', 'Ribosome', 'Secretion system', 'Two-component system']
    
    # Create empty lists for each protein class
    class_lists = {cls: [] for cls in protein_classes}
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        protein_class = row['label']
        if protein_class in protein_classes:
            class_lists[protein_class].append(row.tolist())
    
    # Split each class list into train and test sets
    train_lists = []
    test_lists = []
    for cls in protein_classes:
        train_data, test_data = train_test_split(class_lists[cls], test_size=0.2, random_state=42)
        train_lists.extend(train_data)
        test_lists.extend(test_data)
    
    # Print the lengths of the train and test lists
    print(f"Length of train data: {len(train_lists)}")
    print(f"Length of test data: {len(test_lists)}")
    
    
    # In[2]:
    
    
    train_lists[:2], test_lists[:2]
    
    
    # In[3]:
    
    
    train_words = [item[0] for item in train_lists]
    train_class = [item[2] for item in train_lists]
    
    
    # In[4]:
    
    
    train_words[1000]
    
    
    # In[5]:
    
    
    from sklearn.preprocessing import LabelEncoder
    
    enc = LabelEncoder()
    one_hot_encoded = enc.fit_transform(train_class)
    
    one_hot_encoded_dict = {key: value for key, value in zip(train_words, one_hot_encoded)}
    one_hot_encoded_dict
    
    
    # In[ ]:
    
    
    
    
    
    # In[9]:
    
    
    import os
    import random
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from genomic_embeddings import Embeddings
    import numpy as np
    
    
    model_path = word2vec_model
    gene_embeddings = Embeddings.load_embeddings(model_path)
    
    
    
    
    # Set random seed for reproducibility
    
    sentence_length = 9
    # Randomly choose 1000 files with seed 42
    files = os.listdir(directory)
    random.seed(40)
    selected_files = random.sample(files, 10000)
    
    
    # Define a function to process a file
    def process_file(filename, IP):
        sentences = []
        
        filepath = os.path.join(directory, filename)
    
        with open(filepath, 'r', encoding = "latin1") as file:
            content = file.read()
            words = content.strip().split()
    
            # Check if the key is in the file
            for i in range(4, len(words)-4):
                intermediate = []
                # Shuffle the indices of the words containing the key
                if words[i] == IP:
                    sentence = words[i - 4:i + sentence_length - 4]
                    for j in sentence:
                        try:
                            intermediate.append(gene_embeddings.wv[j])
                        except:
                            continue
                    average = np.mean(intermediate, axis=0)
                    sentences.append(average)
                    
        return sentences
    
    all_embeddings = []
    all_labels = []
    # Iterate over keys
    for IP in tqdm(train_words):
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_file, filename, IP) for filename in selected_files]
            for future in futures:
                sentences = future.result()
                all_embeddings+=sentences
                for _ in range(len(sentences)):
                    all_labels.append(one_hot_encoded_dict[IP])
                
    
    
    
    
    # In[10]:
    
    
    train_words[0]
    
    
    # In[40]:
    
    
    import random
    
    # Zip the lists together
    combined = list(zip(all_embeddings, all_labels))
    
    # Shuffle the combined list
    random.shuffle(combined)
    
    # Unzip the shuffled list
    embeddings, labels = zip(*combined)
    
    
    
    # In[41]:
    
    
    import torch.nn as nn
    
    class Classification_V0(nn.Module):
        def __init__(self, input_dim, first_hidden, second_hidden, last_hidden, output_dim, dropout_prob):
            super(Classification_V0, self).__init__()
            self.fc1 = nn.Linear(input_dim, first_hidden)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Linear(first_hidden, second_hidden)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(second_hidden, last_hidden)
            self.relu3 = nn.ReLU()
            self.fc4 = nn.Linear(last_hidden, output_dim)
            
            self.dropout = nn.Dropout(dropout_prob)
        
        def forward(self, x):
            x = self.fc1(x)
            x = self.relu1(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.relu3(x)
            x = self.dropout(x)
            x = self.fc4(x)
            return x
    
    input_dim = 300
    first_hidden = 128
    second_hidden = 64
    last_hidden = 32
    output_dim = len(protein_classes)
    dropout_prob = 0.20
    
    clf_model = Classification_V0(input_dim, first_hidden, second_hidden, last_hidden, output_dim, dropout_prob)
    
    
    # In[42]:
    
    
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim.lr_scheduler as lr_scheduler
    import torch
    
    batch_size = 1024
    def data_generator(embeddings, labels, batch_size):
        num_samples = len(embeddings)
        for i in range(0, num_samples, batch_size):
            batch_embeddings = embeddings[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            yield batch_embeddings, batch_labels
    
    
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=0.001)
    scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=5, gamma=0.05)
    criterion = nn.CrossEntropyLoss()
    
    
    # In[46]:
    
    
    import numpy as np
    
    num_epochs = 15
    epoch_loss = []
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}:")
        
        # Initialize data generator
        generator = data_generator(embeddings, labels, batch_size)
        train_loss = 0
        # Iterate over batches
        for batch_embeddings, batch_labels in tqdm(generator, desc="Training Batches", leave=False):
            optimizer.zero_grad()
            # Convert data to tensors
            batch_embeddings_tensor = torch.tensor(batch_embeddings, dtype=torch.float32)
    
            batch_labels = np.array(batch_labels)
            batch_labels_tensor = torch.tensor(batch_labels, dtype = torch.long)
    
            outputs = clf_model(batch_embeddings_tensor).squeeze()
            
            loss = criterion(outputs, batch_labels_tensor)
            
            train_loss+=loss.item()
    
            
            loss.backward()
            optimizer.step()
        scheduler.step()
        epoch_loss.append(train_loss/(len(embeddings)/batch_size))
        print(train_loss/(len(embeddings)/batch_size))
    print("Training finished.")
    
    
    # In[50]:
    
    
    train_words = [item[0] for item in test_lists]
    train_class = [item[2] for item in test_lists]
    
    
    # In[51]:
    
    
    one_hot_encoded = enc.transform(train_class)
    
    one_hot_encoded_dict = {key: value for key, value in zip(train_words, one_hot_encoded)}
    
    
    # In[58]:
    
    
    import os
    import random
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from genomic_embeddings import Embeddings
    
    model_path = word2vec_model
    gene_embeddings = Embeddings.load_embeddings(model_path)
    
    
    
    # Set random seed for reproducibility
    
    sentence_length = 9
    files = os.listdir(directory)
    random.seed(42)
    selected_files = random.sample(files, 10000)
    
    
    # Define a function to process a file
    def process_file(filename, IP):
        sentences = []
        
        filepath = os.path.join(directory, filename)
    
        with open(filepath, 'r') as file:
            content = file.read()
            words = content.strip().split()
    
            # Check if the key is in the file
            for i in range(4, len(words)-4):
                intermediate = []
                # Shuffle the indices of the words containing the key
                if words[i] == IP:
                    sentence = words[i - 4:i + sentence_length - 4]
                    for j in sentence:
                        try:
                            intermediate.append(gene_embeddings.wv[j])
                        except:
                            continue
                    average = np.mean(intermediate, axis=0)
                    sentences.append(average)
                    
        return sentences
    
    all_embeddings = []
    all_labels = []
    # Iterate over keys
    for IP in tqdm(train_words):
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = [executor.submit(process_file, filename, IP) for filename in selected_files]
            for future in futures:
                sentences = future.result()
                all_embeddings+=sentences
                for _ in range(len(sentences)):
                    all_labels.append(one_hot_encoded_dict[IP])
                
    
    
    
    
    # In[60]:
    
    
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    
    batch_size = 1024
    generator = data_generator(all_embeddings, all_labels, batch_size)
    
    all_predictions = []
    all_labels = []
    
    for batch_embeddings, batch_labels in tqdm(generator, desc="Evaluation Batches", leave=False):
        batch_embeddings_tensor = torch.tensor(batch_embeddings)
        batch_labels = np.array(batch_labels)
        logits = clf_model(batch_embeddings_tensor)
        predictions = torch.argmax(torch.softmax(logits, dim=-1), dim=-1)
        all_predictions.append(predictions.detach().numpy())
        all_labels.append(batch_labels)
    
    # Concatenate predictions and labels from all batches
    all_predictions = np.concatenate(all_predictions)
    all_labels = np.concatenate(all_labels)
    
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    
    
    # In[61]:
    
    
    # Compute evaluation metrics
    report = classification_report(all_labels, all_predictions, digits=4)
    print("Classification Report:")
    print(report)

if __name__ == '__main__':
    main()



