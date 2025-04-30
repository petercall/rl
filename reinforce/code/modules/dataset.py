#Regular Imports
import torch
from datasets import load_dataset, concatenate_datasets
from torch.utils.data import DataLoader

#File Imports
from .model import tokenizer
from .hyperparams import dataset, system_prompt, batch_size, max_length


#Custom collate_fn function
def collate_wrapper(tokenizer):
    def collate_fn(batch):
        list_of_messages = [text["messages"] for text in batch]
        tokens = tokenizer.apply_chat_template(list_of_messages, tokenize=False, add_generation_prompt=True)    #Make sure to add the generation prompt in!
        x = tokenizer(tokens, return_tensors="pt", truncation = True, padding = True, max_length=max_length)
        return x
    return collate_fn

#Function to get the Bespoke dataset in the correct format
def bespoke(my_dict):
    return {"message" : [{"role" : "system", "content" : system_prompt}, {"role" : "user", "content" : my_dict["messages"][0]["content"]}]}

#Load in the dataset and apply the function to it
data = load_dataset(dataset, split = "train")
data = data.map(bespoke, remove_columns=["system", "conversations", "messages"])
data = data.rename_column("message", "messages")    #Final column is called "messages"

#Split the dtatset into a train and test dataset
dataset = data.train_test_split()
train_dataset = dataset["train"]
validation_dataset = dataset["test"]

#Define the train loader and the validation loader
train_loader = DataLoader(train_dataset, collate_fn = collate_wrapper(tokenizer), shuffle = True, batch_size = batch_size, num_workers = 4)
val_loader = DataLoader(validation_dataset, collate_fn = collate_wrapper(tokenizer), batch_size = batch_size, num_workers = 4)