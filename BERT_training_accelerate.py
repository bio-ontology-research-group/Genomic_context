#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.optim as optim
import random
from tqdm import tqdm
import re
from pynvml import *
import psutil
from accelerate import Accelerator
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import AutoModelForMaskedLM, BertConfig, get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import os
import wandb


# In[2]:


def main():
    accelerator = Accelerator() 


    wandb.login(key = "5c0f1505d0af16a0dda3f3d031310d45e9a3f07b")

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer_path = "WordLevel_tokenizer_trained_InterPro.json"
    tokenizer = tokenizer.from_file(tokenizer_path)
    tokenizer.enable_truncation(256)
    train_dataset = Dataset.load_from_disk('BERT_train_dataset_context5_no_pad')
    val_dataset = Dataset.load_from_disk('BERT_val_dataset_context5_no_pad')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False)
    accelerator.print("Len of train_dataloader: ",len(train_dataloader))
    config = BertConfig(vocab_size = tokenizer.get_vocab_size(), hidden_size = 256, num_hidden_layers = 4, num_attention_heads = 8, intermediate_size = 256)
    model = AutoModelForMaskedLM.from_config(config)

    epochs = 7
    optimizer = optim.AdamW(model.parameters(),lr=1e-3, weight_decay=2e-5)

    num_training_steps = epochs * len(train_dataloader) 
    num_warmup_steps = int(num_training_steps*0.05)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    def print_gpu_utilization():
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(0)
        info = nvmlDeviceGetMemoryInfo(handle)
        accelerator.print(f"GPU memory occupied: {info.used//1024**2} MB.")


    # Function to get free CPU memory
    def get_free_memory():
        memory = psutil.virtual_memory()
        return memory.available / (1024.0 ** 3)  # Convert bytes to gigabytes

    # Display free CPU memory
    accelerator.print(f"Free CPU Memory: {get_free_memory():.2f} GB")
    wandb.init(
    # set the wandb project where this run will be logged
    project="InterPro_BERT_training_context5_no_pad",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 1e-3,
    "architecture": "BERT",
    "dataset": "InterPro_genomes",
    "epochs": 10,
    }
    )
    accelerator.print("LOADING MODEL")
    model, optimizer, scheduler, train_dataloader = accelerator.prepare(model,optimizer, scheduler, train_dataloader)
    accelerator.print(len(train_dataloader))
    accelerator.print("NOW WILL START TRAINING")
    training_loss = []
    validation_loss = []
    val_acc = []

    # best_val_loss = float('inf')  
    # patience = 3 
    for epoch in tqdm(range(epochs)):
        total_correct = 0
        total_tokens = 0
        train_loss = 0
        val_loss = 0
        model.train()
        accelerator.print(f"training epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids,attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            train_loss+=loss.item()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            if step%10000==0:
                wandb.log({"train_loss": loss.item()})
        print_gpu_utilization()
        accelerator.print(f"evaluation epoch {epoch}")
        model.eval()
        count=0
        for step, batch in enumerate(val_dataloader):
            input_ids = batch['input_ids'].to(accelerator.device)
            attention_mask = batch['attention_mask'].to(accelerator.device)
            labels = batch['labels'].to(accelerator.device)
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=-1)
            # Mask out labels where input_ids != 4
            mask = (input_ids == 4)
            masked_labels = labels[mask]
            masked_predicted_labels = predicted_labels[mask]

            correct = torch.sum(masked_predicted_labels == masked_labels).item()
            total_correct += correct
            total_tokens += masked_labels.numel()

            val_loss += loss.item()
            if step%5000==0:
                wandb.log({"val_loss":loss.item(), "val_acc":correct/masked_labels.numel()})
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

        avg_train_loss = train_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)

        training_loss.append(avg_train_loss)
        validation_loss.append(avg_val_loss)

        accelerator.print("Train loss:", avg_train_loss)
        accelerator.print("Val loss:", avg_val_loss)
        accelerator.print("Acc: ", accuracy )
        accelerator.print("\n\n")

    #     if avg_val_loss < best_val_loss:
    #         best_val_loss = avg_val_loss
    #         torch.save(model.state_dict(), '/BERT_context_pretrained_10K/BERT_best.pth')  # Save the best model

    #     else:
    #         patience -=1
    #         if patience== 0:
    #             # Stop training if validation loss doesn't improve after patience epochs
    #             print(f"Stopping early as validation loss didn't improve for {patience} epochs.")
    #             break  # Break out of the training loop


    wandb.finish()
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        'BERT_context_pretrained_InterPro_final_context5_no_pad',
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save)
    print("Saved pre_trained model here: BERT_context_pretrained_InterPro_final_context5_no_pad")

    if accelerator.is_main_process:
        plt.plot(list(range(len(training_loss))), training_loss, linestyle='dotted', label='Training Loss')
        plt.plot(list(range(len(validation_loss))), validation_loss, marker='o', linestyle='solid', label='Validation Loss')

        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()


        plt.savefig('loss_plot_final_context5_no_pad.png', dpi=300) 



# In[3]:


if __name__=="__main__":
    main()


# In[ ]:




