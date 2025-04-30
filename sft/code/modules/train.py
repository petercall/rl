#Regular imports
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
      
#Define a validation function
@torch.no_grad()
def validate(model, loss_func, device, val_loader):
   model.eval()
   val_loop_losses = []
  
   for x, y in val_loader:
        #Add them to the gpu
        x = {name : tens.to(device) for name, tens in x.items()}
        y = y.to(device)
            
        #Run the model
        output = model(**x)     #Note: You could add model_kwargs as one of the inputs to the validate() function and then add it as: model(**x, **model_kwargs)  
        
        #Calculate and add the loss to the list
        b, s, v = output.logits.shape   #output.logits is of shape (batch_size, seq_len, vocab_size)
        loss = loss_func(output.logits.view(b*s,v), y.view(-1)) #We need to input something of size x = (n, vocab_size) and y = (n), so we reshape to make that happen
        val_loop_losses.append(loss.item())
        
   return np.mean(val_loop_losses)
      
#Define a training loop function
def train(model, optimizer, scheduler, loss_func, device, train_loader, val_loader, validate, epochs, val_interval, smallest_val_loss, patience, checkpoint_filepath):
    train_losses = []
    val_losses = []
    patience_count = 0
    val_epoch_count = 0
    
    for epoch in tqdm(range(epochs)):
        for x, y in train_loader:
            #Add them to the gpu
            x = {name : tens.to(device) for name, tens in x.items()}
            y = y.to(device)
    
            #Zero out the gradients
            optimizer.zero_grad()

            #Run the model
            output = model(**x)     #Note: You could add model_kwargs as one of the inputs to the train() function and then add it as: model(**x, **model_kwargs)  
            
            #Calculate and add the loss to the list
            b, s, v = output.logits.shape   #output.logits is of shape (batch_size, seq_len, vocab_size)
            loss = loss_func(output.logits.view(b*s,v), y.view(-1)) #We need to input something of size x = (n, vocab_size) and y = (n), so we reshape to make that happen
            train_losses.append(loss.item())
            
            #Run the backward pass and take the step
            loss.backward()
            optimizer.step()

        if epoch % val_interval == 0:
            val_epoch_count += 1
            val_loss = validate(model, loss_func, device, val_loader)
            val_losses.append(val_loss)

            print(f"Epoch: {epoch+1}/{epochs}, Training Loss: {round(loss.item(),3)}, Validation Loss: {round(val_loss,3)}")

            if val_loss < smallest_val_loss:
                patience_count = 0     
                smallest_val_loss = val_loss
                checkpoint_dict = {
                    "epoch" : epoch,
                    "model_state" : model.state_dict(),
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