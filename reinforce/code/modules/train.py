#Regular imports
import torch
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
      
      
      
#Define the function that will calculate the sum of the log probabilities of the tokens that were generated
def compute_log_probs(model, id_tensor, pad_token_id):
    #NOTE: This function has a HUGE omission in it in that it does NOT mask out the log probabilities of the id's in the id_tensor that are from the input sequence.
    #We only want to consider the ids from the model's answer, but it is tricky to differentiate the input questions differ in length so the spot where the model's answer beings differs for each row.
    #We would need to use the original x["input_ids"] and x["attention_mask"] to figure it out, but it is non-trivial.
    #It does properly mask out log probabilites in id_tensor that are padding, however.
    """
    Compute the log probabilities of the generated tokens within the id_tensor.
    
    Args:
        model: LLM model
        id_tensor: tensor of shape (batch_size, seq_len) containing both input ids and generated ids
        pad_token_id: the padding token id
        
    Returns:
        log_probs_sum: tensor of shape (batch_size,) with summed log probabilities for each generated sequence
    """

    # Get model outputs for the full id_tensor (input + generated tokens)
    outputs = model(id_tensor)
    logits = outputs.logits  # Shape: (batch_size, seq_len, vocab_size)

    # Shift logits and labels for language modeling
    logits = logits[:, :-1, :].contiguous()  # Remove the last token (no next-token prediction)
    labels = id_tensor[:, 1:].contiguous()  # Shift labels by 1 for next-token prediction

    # Gather log-probs for the actual generated tokens (next-token prediction)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Select log probs for the actual generated tokens using the labels
    generated_log_probs = log_probs.gather(2, labels.unsqueeze(2)).squeeze(2)

    # Create a mask to zero out padding tokens and the input tokens
    mask = torch.zeros_like(labels, dtype=torch.bool)  # Shape: (batch_size, seq_len-1)
    
    # Identify non-padding tokens in the generated sequence
    non_pad_mask = labels != pad_token_id  # Mask for non-padding tokens
    
    # Apply the mask to only keep log-probs for non-padding tokens
    generated_log_probs = generated_log_probs * non_pad_mask.float()

    # Sum the log probabilities over the valid generated tokens (non-padding tokens)
    log_probs_sum = generated_log_probs.sum(dim=1).squeeze()  # Shape: (batch_size,)

    return log_probs_sum

#This function adds the tensor <extra_0> at each think step (which is denoted by \n\n) of the model's output
def add_extra(input_tensor, extra_id, enter_id):
    mask = input_tensor == enter_id
    input_tensor[mask] = extra_id
    return input_tensor

#This calculates the reward from the logits and token_mask, as being the probabilities tied to the <extra_0> tokens
def make_step_rewards(logits, token_masks):
    print(token_masks.shape)
    print(logits.shape)
    
    probabilities = F.softmax(logits, dim=-1)
    print(probabilities.shape)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

#Define a validation function
@torch.no_grad()
def validate(model, reward_model, generation_args, device, val_loader):
   model.eval()
   val_loop_losses = []
  
   for x in val_loader:
        #Add them to the gpu
        x = {name : tens.to(device) for name, tens in x.items()}

        #Run the model
        output = model.generate(**x, **generation_args)     #output is (batch_size, seq_len) and the tokens in every row include both the input AND the model's response
        
        #Add in the <extra_0> tokens
        output = add_extra(output, extra_id, enter_id)
        
        #Calculate the reward of what the model output
        reward_output = reward_model(output)
        token_mask = ((x["input_ids"] == pad_id) | (x["input_ids"] == extra_id))
        step_rewards = make_step_rewards(reward_output.last_hidden_state, token_mask)
        rewards = step_rewards.mean(-1).squeeze()
        
        #Calculate the sum of the log probabilities of the tokens that were generated
        log_prob_sum = compute_log_probs(model, output, pad_token_id)
        
        #Calculate and add the loss to the list
        loss = -(log_prob_sum * rewards).mean()        
        val_loop_losses.append(loss.item())
        
   return np.mean(val_loop_losses)
      
#Define a training loop function
def train(model, generation_args, reward_model, pad_id, extra_id, enter_id, optimizer, scheduler, device, train_loader, val_loader, validate, epochs, val_interval, smallest_val_loss, patience, checkpoint_filepath):
    train_losses = []
    val_losses = []
    patience_count = 0
    val_epoch_count = 0
    
    for epoch in tqdm(range(epochs)):
        for x in train_loader:
            #Add them to the gpu
            x = {name : tens.to(device) for name, tens in x.items()}
             
            #Zero out the gradients
            optimizer.zero_grad()

            #Run the model
            output = model.generate(**x, **generation_args)     #output is (batch_size, seq_len) and the tokens in every row include both the input AND the model's response
            
            #Add in the <extra_0> tokens
            output = add_extra(output, extra_id, enter_id)
            
            #Calculate the reward of what the model output
            reward_output = reward_model(output)
            token_mask = ((x["input_ids"] == pad_id) | (x["input_ids"] == extra_id))
            step_rewards = make_step_rewards(reward_output.last_hidden_state, token_mask)
            rewards = step_rewards.mean(-1).squeeze()
            
            #Calculate the sum of the log probabilities of the tokens that were generated
            log_prob_sum = compute_log_probs(model, output, pad_token_id)
            
            #Calculate and add the loss to the list
            loss = -(log_prob_sum * rewards).mean()
            train_losses.append(loss.item())
            
            #Run the backward pass and take the step
            loss.backward()
            optimizer.step()

        if epoch % val_interval == 0:
            val_epoch_count += 1
            val_loss = validate(model, reward_model, generation_args, device, val_loader)
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