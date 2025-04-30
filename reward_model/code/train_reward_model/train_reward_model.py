#Regular imports
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

#File imports
from reward_model_hyperparams import epochs, val_interval, smallest_val_loss, stop_patience, checkpoint_load_location, checkpoint_storage_location, graph_save_location, scheduler_patience, scheduler_factor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from modules import reward_model, tokenizer, train_loader, val_loader, train, validate, graph_losses




#Change to the current working directory to ensure relative imports work properly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Download the checkpoint and then the model
checkpoint = torch.load(checkpoint_load_location)
reward_model.load_state_dict(checkpoint["reward_model_state"])

#Create the optimizer and lr_scheduler
optimizer = optim.AdamW(reward_model.parameters())
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience = scheduler_patience, factor = scheduler_factor)

# Train the model
train_losses, val_losses, smallest_val_loss = train(
    reward_model,       #model
    optimizer,          #optimizer
    scheduler,          #lr scheduler
    reward_model.base_model.device,  #torch device
    train_loader,       #train loader
    val_loader,         #val loader
    validate,           #function used for validation
    epochs,             #epochs to train for
    initial_val,        #Whether to get a validation loss before starting training
    val_interval,       #How often to validate
    smallest_val_loss,  #current smallest validation loss
    stop_patience,      #Number of validation runs with no improvement after which we terminate the training loop
    checkpoint_storage_location,    #location to store checkpoint
    )

print(f"Smallest Val loss is: {smallest_val_loss}")

# #Save graphs of the losses
graph_losses(train_losses, "Train", graph_save_location)
graph_losses(val_losses, "Validation", graph_save_location)
