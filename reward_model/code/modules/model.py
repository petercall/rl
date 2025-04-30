#Regular imports
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

#File imports
from .hyperparams import model_name, model_args, special_tokens_to_add, embedding_scaler


#Define the RewardModel class
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super(RewardModel, self).__init__()
        self.base_model = base_model
        self.prediction_head = nn.Linear(base_model.config.hidden_size, 1, dtype=base_model.dtype)
        
    def forward(self, input_ids, attention_mask = None, **kwargs):
        #Pass the inputs through the base model
        base_output = self.base_model(input_ids, attention_mask = attention_mask, **kwargs)
        
        #Get the final embedding that is NOT associated with a padding vector
        batch_size, seq_len, hidden_size = base_output.last_hidden_state.shape  
        if attention_mask is not None:
            final_positions = attention_mask.sum(dim=1) - 1
        else:
            final_positions = torch.full((batch_size,), seq_len - 1, device=input_ids.device)
        final_embeddings = base_output.last_hidden_state[torch.arange(batch_size, device = input_ids.device), final_positions] #final_embeddings of shape (batch_size, hidden_size)
        
        #Pass the final embeddings through the prediction head and squeeze to make a 1D vector
        prediction_output = self.prediction_head(final_embeddings).squeeze()    #The output was (batch_size, 1), now it is (batch_size)
        
        return prediction_output


#Download the base model and tokenizer
base_model = AutoModel.from_pretrained(model_name, **model_args)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Add the special tokens and expand the base model vocabulary
tokenizer.add_special_tokens(special_tokens_to_add)
base_model.resize_token_embeddings(len(tokenizer), mean_resizing = False)

#Set the initial embeddings of the added tokens to be like eos_token or bos_token plus some random noise
with torch.no_grad():
    for token in special_tokens_to_add["additional_special_tokens"]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if "/" in token:
            current_id = tokenizer.eos_token_id
        else:
            current_id = tokenizer.bos_token_id
        current_embedding = base_model.embed_tokens.weight[current_id].clone()
        base_model.embed_tokens.weight[token_id] = current_embedding + embedding_scaler*torch.randn_like(current_embedding)

#Set the initial embeddings of the pad token to be 0
with torch.no_grad():
    base_model.embed_tokens.weight[tokenizer.pad_token_id] = torch.zeros_like(base_model.embed_tokens.weight[tokenizer.pad_token_id])

#Use .register_hooks() to ensure that the pad token is not updated during training
def zero_pad_grad(pad_id):
    def hook(grad):
        grad[pad_id] = torch.zeros_like(grad[pad_id])
        return grad
    return hook
base_model.embed_tokens.weight.register_hook(zero_pad_grad(tokenizer.pad_token_id))   
    
#Intantiate the reward model
reward_model = RewardModel(base_model)
reward_model.prediction_head.to(base_model.device)