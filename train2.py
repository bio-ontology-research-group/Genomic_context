from pathlib import Path
import click as ck
from transformers import (
    TrainingArguments, Trainer, AutoConfig
)
from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import evaluate
import numpy as np
import os
from tqdm import tqdm
os.environ["TOKENIZERS_PARALLELISM"] = "true"

metric = evaluate.load("accuracy")
accelerator  = Accelerator()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


@ck.command()
def main():
    training_args = TrainingArguments(
        output_dir="interpro_bert_2",
        learning_rate=1e-4,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=128,
        num_train_epochs=15,
        weight_decay=0.01,
        save_strategy = "no",
        evaluation_strategy="epoch",
        push_to_hub=True,
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file='interpro_tokenizer.json')
    tokenizer.add_special_tokens({
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_token': '<unk>',
        'mask_token': '<mask>'
    })
    context_length = 20
    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_tensors= "pt"
        )
        
        return {"input_ids": outputs["input_ids"]}

    dataset = load_dataset("text",
        data_files={'train': 'train.txt',
                            'test': 'test.txt'})
    
    outputs = tokenizer(
        dataset["train"][:2]["text"],
        truncation=True,
        max_length=context_length,
        return_tensors= "pt"
    )
    accelerator.print("Input sentences: ", dataset["train"][:2])
    accelerator.print("Encoded inputs: ", outputs)


    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.20)
    
    dataset = dataset.map(tokenize, batched=True, remove_columns=dataset["train"].column_names)
    accelerator.print("Dataset[0] example: ", dataset['train'][0])

    config = BertConfig(
        vocab_size=len(tokenizer), num_hidden_layers=12, hidden_size=512, num_attention_heads=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id)
    model = BertForMaskedLM(config)
    model_size = sum(t.numel() for t in model.parameters())
    accelerator.print(f"Bert size: {model_size/1000**2:.1f}M parameters")
    accelerator.print(model)
    trainer = accelerator.prepare(Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics,
    ))
    accelerator.print("Starting training process")
    trainer.train()
    if accelerator.is_main_process:
        trainer.push_to_hub()
        print("Model successfully pushed to the HUB")
        trainer.save_model("InterPro_BERT_2")
        print("Training done successfully")
if __name__ == '__main__':
    main()
