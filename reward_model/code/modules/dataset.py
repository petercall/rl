#Regular Imports
import re
import torch.nn as nn
from datasets import load_dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

#File Imports
from .model import tokenizer
from .hyperparams import batch_size


#Function to input the special tokens
def convert_to_tokenized_format(row_dicts):
    names = ["chosen", "rejected"]
    return_dict = dict()
    return_names = ["chosen_tokens", "rejected_tokens"]
    
    for j in range(2):
        text = row_dicts[names[j]]
        # Split on double newlines and clean extra whitespace
        segments = [seg.strip() for seg in re.split(r'\n{2,}', text) if seg.strip()]

        formatted = []
        for i in range(0, len(segments), 2):
            # Handle incomplete pairs safely
            if i + 1 >= len(segments):
                break

            human = re.sub(r'^Human:\s*', '', segments[i], flags=re.IGNORECASE)
            assistant = re.sub(r'^Assistant:\s*', '', segments[i+1], flags=re.IGNORECASE)

            # For all but the last turn, format normally
            if i + 2 < len(segments):
                formatted.append(f"<|user|>{human}<|/user|><|assistant|>{assistant}<|/assistant|>")
            else:
                # Final assistant message should be wrapped in <|evaluate|> block
                formatted.append(f"<|user|>{human}<|/user|><|eval|><|assistant|>{assistant}<|/assistant|><|/eval|>")
        return_dict[return_names[j]] = " ".join(formatted)

    return return_dict

#Custom collate_fn function
def collate_wrapper(tokenizer):
    def collate_fn(batch):
        chosen = [text["chosen_tokens"] for text in batch]
        rejected = [text["rejected_tokens"] for text in batch]
        chosen = tokenizer(chosen, return_tensors="pt", truncation = True, padding = True)
        rejected = tokenizer(rejected, return_tensors="pt", truncation = True, padding = True)
        return {"chosen" : chosen, "rejected" : rejected}
    return collate_fn


#Load in the data
data = load_dataset("Anthropic/hh-rlhf")
data = data.map(convert_to_tokenized_format, remove_columns=["chosen", "rejected"])

#Split into train and validation datasets. The column names of the datasets are: "chosen" and "rejected"
train_dataset = data["train"]
val_dataset = data["test"]

#Define the train loader and the validation loader
train_loader = DataLoader(train_dataset, collate_fn = collate_wrapper(tokenizer), shuffle = True, batch_size = batch_size, pin_memory = True, num_workers = 4)
val_loader = DataLoader(val_dataset, collate_fn = collate_wrapper(tokenizer), batch_size = batch_size, pin_memory = True, num_workers = 4)