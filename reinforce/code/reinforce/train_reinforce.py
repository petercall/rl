#Regular Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn import functional as F
import os
import sys

#File Imports
from hyperparams import epochs, val_interval, smallest_val_loss, stop_patience, checkpoint_storage_location, graph_save_location, scheduler_patience, scheduler_factor, extra_id, enter_id, generation_args
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules import model, reward_model, tokenizer, train_loader, val_loader, train, validate, graph_losses



#Change to the current working directory to ensure relative imports work properly
os.chdir(os.path.dirname(os.path.abspath(__file__)))


#Define the optimizer
optimizer = optim.Adam(model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = scheduler_patience, factor = scheduler_factor)

#Add the tokenizer eos token id to the generation args
generation_args["eos_token_id"] = tokenizer.eos_token_id

# Train the model
train_losses, val_losses, smallest_val_loss = train(
    model,              #model
    generation_args,    #generation arguments inside the training loop
    reward_model,       #reward_model
    tokenizer.pad_token_id, #pad token id
    extra_id,               #token id for <extra_0>
    enter_id,               #token id for \n\n
    optimizer,          #optimizer
    scheduler,          #lr scheduler
    model.device,       #torch device
    train_loader,       #train loader
    val_loader,         #val loader
    validate,           #function used for validation
    epochs,             #epochs to train for
    val_interval,       #How often to validate
    smallest_val_loss,  #current smallest validation loss
    stop_patience,      #Number of validation runs with no improvement after which we terminate the training loop
    checkpoint_storage_location,    #location to store checkpoint
    )

print(f"Smallest Val loss is: {smallest_val_loss}")

# #Save graphs of the losses
graph_losses(train_losses, "Train", graph_save_location)
