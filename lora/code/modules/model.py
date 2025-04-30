#Regular imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model

#File imports
from .hyperparams import model_name, model_args, lora_config

#Download the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Setup the Lora configuration
model = get_peft_model(model, lora_config)

print(model.print_trainable_parameters())