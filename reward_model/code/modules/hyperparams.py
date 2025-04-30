import torch

#Model hyperparameters
model_name = "meta-llama/Llama-3.2-1B"
dtype = torch.bfloat16
model_args = {"torch_dtype" : dtype, "device_map" : "auto"}

#Tokenizer hyperparameters
special_tokens_to_add = {
    "pad_token" : "<|pad|>",
    "additional_special_tokens" : [
        "<|user|>", 
        "<|/user|>", 
        "<|assistant|>", 
        "<|/assistant|>", 
        "<|eval|>", 
        "<|/eval|>"
    ]
}
embedding_scaler = 0.02

#Data Loader hyperparameters
batch_size = 8
