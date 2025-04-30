#Regular Imports
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

#File Imports
from .model import tokenizer
from .hyperparams import datasets, system_prompt, batch_size, max_length


#Custom collate_fn function
def collate_wrapper(tokenizer):
    def collate_fn(batch):
        list_of_messages = [text["messages"] for text in batch]
        tokens = tokenizer.apply_chat_template(list_of_messages, tokenize=False)
        x = tokenizer(tokens, return_tensors="pt", truncation = True, padding = True, max_length=max_length)
        y = torch.roll(x["input_ids"], shifts = -1, dims = 1)
        y[:,-1] = tokenizer.pad_token_id
        return x, y
    return collate_fn

#Function to get the OpenThoughts dataset in the correct format
def open_thoughts(my_dict):
    return {"messages" : [{"role" : "system", "content" : system_prompt}, {"role" : "user", "content" : my_dict["conversations"][0]["value"]}, {"role" : "assistant", "content" : my_dict["conversations"][1]["value"]}]}

#Function to get the Bespoke dataset in the correct format
def bespoke(my_dict):
    return {"message" : [{"role" : "system", "content" : system_prompt}, {"role" : "user", "content" : my_dict["messages"][0]["content"]}, {"role" : "assistant", "content" : my_dict["messages"][1]["content"]}]}


#Load in the dataset
data1 = load_dataset(datasets[0], split = "train")
data2 = load_dataset(datasets[1], split = "train")

#Get the datasets in the correct format
data1 = data1.map(open_thoughts, remove_columns=["system", "conversations"])
data2 = data2.map(bespoke, remove_columns=["system", "conversations", "messages"])

#Rename the "message" column to "messages" in data2, and concatenate the datasets
data2 = data2.rename_column("message", "messages")
dataset = concatenate_datasets([data1, data2])

#Split the dtatset into a train and test dataset
dataset = dataset.train_test_split()
train_dataset = dataset["train"]
validation_dataset = dataset["test"]

#Define the train loader and the validation loader
train_loader = DataLoader(train_dataset, collate_fn = collate_wrapper(tokenizer), shuffle = True, batch_size = batch_size, num_workers = 4)
val_loader = DataLoader(validation_dataset, collate_fn = collate_wrapper(tokenizer), batch_size = batch_size, num_workers = 4)
