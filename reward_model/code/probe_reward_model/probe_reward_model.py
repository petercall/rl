#Regular Imports
import os
import sys
import torch

#File Imports
from hyperparams import questions_to_ask, fully_trained_checkpoint_location, prediction_head_checkpoint_location               
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules import reward_model, tokenizer


#Set the device
device = reward_model.base_model.device

#Get rewards from the untrained model
print("Rewards for Untrained Model")
for i, question in enumerate(questions_to_ask):
    texts = [question["chosen"], question["rejected"]]
    inputs = tokenizer(texts, return_tensors = "pt", padding = True)
    inputs = {name : tens.to(device) for name, tens in inputs.items()}
    values = reward_model(**inputs)
    print(f"\tQuestion {i+1} -- Chosen: {values[0].item()}, Rejected: {values[1].item()}")


#Load in the prediction-head-trained model
checkpoint = torch.load(prediction_head_checkpoint_location)
reward_model.load_state_dict(checkpoint["reward_model_state"])

#Get rewards from the prediction-head-trained model
print()
print("Rewards for Prediction-head-trained Model")
for i, question in enumerate(questions_to_ask):
    texts = [question["chosen"], question["rejected"]]
    inputs = tokenizer(texts, return_tensors = "pt", padding = True)
    inputs = {name : tens.to(device) for name, tens in inputs.items()}
    values = reward_model(**inputs)
    print(f"\tQuestion {i+1} -- Chosen: {values[0].item()}, Rejected: {values[1].item()}")


#Load in the fully-trained model
checkpoint = torch.load(fully_trained_checkpoint_location)
reward_model.load_state_dict(checkpoint["reward_model_state"])

# Get rewards from the fully-trained model
print()
print("Rewards for Fully-trained Model")
for i, question in enumerate(questions_to_ask):
    texts = [question["chosen"], question["rejected"]]
    inputs = tokenizer(texts, return_tensors = "pt", padding = True)
    inputs = {name : tens.to(device) for name, tens in inputs.items()}
    values = reward_model(**inputs)
    print(f"\tQuestion {i+1} -- Chosen: {values[0].item()}, Rejected: {values[1].item()}")