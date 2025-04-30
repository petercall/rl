#Regular imports
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
      
#Define a validation function
@torch.no_grad()
def validate(reward_model, device, val_loader):
    reward_model.eval()
    val_loop_losses = []
  
    for batch in val_loader:
        #Get the chosen and rejected dictionaries and put their tensors on the GPU
        chosen = batch["chosen"]
        rejected = batch["rejected"]
        chosen = {name : tens.to(device, non_blocking = True) for name, tens in chosen.items()}
        rejected = {name : tens.to(device, non_blocking = True) for name, tens in rejected.items()}
            #chosen and rejected are of shape: (batch_size, seq_len)
        
        #Push through the reward function and caluclate the loss
        reward_diff = reward_model(**chosen) - reward_model(**rejected)  
        loss = -F.logsigmoid(reward_diff).mean()
        val_loop_losses.append(loss.item())
    
    model.train()    
    return np.mean(val_loop_losses)
      
#Define a training loop function
def train(reward_model, optimizer, scheduler, device, train_loader, val_loader, validate, epochs, initial_val, val_interval, smallest_val_loss, patience, checkpoint_filepath):
    train_losses = []
    val_losses = []
    patience_count = 0
    
    #Get an initial validation loss if desired
    if initial_val:
        val_loss = validate(model, loss_func, device, val_loader)
        val_losses.append(val_loss)
        print(f"Initial Validation Loss: {round(val_loss,3)}", flush = True)
    
    for epoch in tqdm(range(epochs)):
        for batch in train_loader:
            #Get the chosen and rejected dictionaries and put their tensors on the GPU
            chosen = batch["chosen"]
            rejected = batch["rejected"]
            chosen = {name : tens.to(device, non_blocking = True) for name, tens in chosen.items()}
            rejected = {name : tens.to(device, non_blocking = True) for name, tens in rejected.items()}
                #chosen and rejected are of shape: (batch_size, seq_len)
    
            #Zero out the gradients
            optimizer.zero_grad()

            #Run the model
            reward_diff = reward_model(**chosen) - reward_model(**rejected)  
            loss = -F.logsigmoid(reward_diff).mean()
            train_losses.append(loss.item())
            
            #Run the backward pass and take the step
            loss.backward()
            optimizer.step()


        if epoch % val_interval == 0:
            val_loss = validate(reward_model, device, val_loader)
            val_losses.append(val_loss)

            print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {round(loss.item(),3)}, Validation Loss: {round(val_loss,3)}", flush = True)

            if val_loss < smallest_val_loss:
                patience_count = 0     
                smallest_val_loss = val_loss
                checkpoint_dict = {
                    "epoch" : epoch,
                    "reward_model_state" : reward_model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scheduler_state" : scheduler.state_dict()
                }
                torch.save(checkpoint_dict, checkpoint_filepath)
            else:
                patience_count += 1
                if patience_count >= patience:
                    return train_losses, val_losses, smallest_val_loss
                
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()

    return train_losses, val_losses, smallest_val_loss    