#Regular imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

#File imports
from .hyperparams import model_name, model_args

#Download the base model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
tokenizer = AutoTokenizer.from_pretrained(model_name)
