import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
import numpy as np
import os

# Define the dataset
class TokenizationDataset(Dataset):
    def __init__(self, tokenizer, texts, tokenized_texts):
        self.tokenizer = tokenizer
        self.texts = texts
        self.tokenized_texts = tokenized_texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        source = self.texts[idx]
        target = self.tokenized_texts[idx]

        source_tokenized = self.tokenizer.encode_plus(
            text = 'tokenize the word: ' + source, 
            max_length=40,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        target_tokenized = self.tokenizer.encode_plus(
            target, 
            max_length=30,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        source_ids = source_tokenized['input_ids'].squeeze()
        target_ids = target_tokenized['input_ids'].squeeze()
        
        return {"input_ids": source_ids, "labels": target_ids}

train_src_path = "dataset/tar.train.src"
train_src = np.loadtxt(train_src_path, dtype=str)
train_tgt_path = "dataset/tar.train.tgt"
train_tgt = np.loadtxt(train_tgt_path, dtype=str, delimiter='\t')

dev_src_path = "dataset/tar.dev.src"
dev_src = np.loadtxt(dev_src_path, dtype=str)
dev_tgt_path = "dataset/tar.dev.tgt"
dev_tgt = np.loadtxt(dev_tgt_path, dtype=str, delimiter='\t')

# Load a pre-trained T5 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('google/byt5-xl', torch_dtype=torch.bfloat16)
model = T5ForConditionalGeneration.from_pretrained('google/byt5-xl', torch_dtype=torch.bfloat16)

# Prepare the dataset
train_dataset = TokenizationDataset(tokenizer, train_src, train_tgt)
dev_dataset = TokenizationDataset(tokenizer, dev_src, dev_tgt)


# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='/data/user_data/zw3/subword_results',
    num_train_epochs=100,
    per_device_train_batch_size=32,
    bf16=True,
    bf16_full_eval=True,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    save_strategy='epoch',
    learning_rate =1e-4,
    save_total_limit=3,
    evaluation_strategy='epoch',  # Perform evaluation at the end of each epoch
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    # callbacks = [EarlyStoppingCallback(early_stopping_patience=5)]
)

# Train and evaluate the model
trainer.train()

test_src_path = "dataset/tar.dev.src"
test_src = np.loadtxt(test_src_path, dtype=str)
test_src = ['tokenize the word: '+ s for s in test_src.tolist()]

# Tokenize the test_src data
tokenized_test_src = tokenizer(
    test_src, 
    max_length=40, 
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)

model.to(trainer.args.device) 

# Generate predictions
input_ids = tokenized_test_src["input_ids"].to(trainer.args.device)  # Ensure input_ids is on the same device as the model

with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, num_beams=4)

# Decode the generated token IDs
decoded_predictions = []
for output in outputs:
    decoded_prediction = tokenizer.decode(output, skip_special_tokens=True)
    decoded_predictions.append(decoded_prediction)

print(decoded_predictions)

test_src_path = "dataset/tar.test.src"
test_src = np.loadtxt(test_src_path, dtype=str)
test_src = ['tokenize the word: '+ s for s in test_src.tolist()]

# Tokenize the test_src data
tokenized_test_src = tokenizer(
    test_src, 
    max_length=40, 
    padding='max_length', 
    truncation=True, 
    return_tensors="pt"
)


# Generate predictions
input_ids = tokenized_test_src["input_ids"].to(trainer.args.device)  # Ensure input_ids is on the same device as the model
model.to(trainer.args.device) 
with torch.no_grad():
    outputs = model.generate(input_ids=input_ids, num_beams=4)

# Decode the generated token IDs
decoded_predictions = []
for output in outputs:
    decoded_prediction = tokenizer.decode(output, skip_special_tokens=True)
    decoded_predictions.append(decoded_prediction)

print(decoded_predictions)