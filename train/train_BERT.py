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
accelerator = Accelerator()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

@ck.command()
@ck.option('--tokenizer-path', default='interpro_tokenizer.json', help='Path to the tokenizer file')
@ck.option('--output-dir', default='InterPro_BERT', help='Output directory for the model')
@ck.option('--push-to-hub', is_flag=True, default=False, help='Whether to push the model to Hugging Face Hub')
@ck.option('--epochs', default=15, type=int, help='Number of training epochs')
@ck.option('--mlm-probability', default=0.20, type=float, help='MLM probability')
@ck.option('--train-batch-size', default=256, type=int, help='Per device train batch size')
@ck.option('--eval-batch-size', default=128, type=int, help='Per device evaluation batch size')
@ck.option('--save-strategy', type=ck.Choice(['no', 'epoch']), default='no', help='Save strategy for the model')
def main(tokenizer_path, output_dir, push_to_hub, epochs, mlm_probability, train_batch_size, eval_batch_size, save_strategy):
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=1e-4,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        save_strategy=save_strategy,
        evaluation_strategy="epoch",
        push_to_hub=push_to_hub,
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
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
            return_tensors="pt"
        )
        return {"input_ids": outputs["input_ids"]}

    dataset = load_dataset("text",
        data_files={'train': 'train.txt',
                    'test': 'test.txt'})
    
    outputs = tokenizer(
        dataset["train"][:2]["text"],
        truncation=True,
        max_length=context_length,
        return_tensors="pt"
    )
    accelerator.print("Input sentences: ", dataset["train"][:2])
    accelerator.print("Encoded inputs: ", outputs)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=mlm_probability)
    
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
        if push_to_hub:
            trainer.push_to_hub()
            print("Model successfully sent to the HF")
        trainer.save_model(output_dir)
        print("Training done successfully")

if __name__ == '__main__':
    main()
