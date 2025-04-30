#Regular imports
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import get_peft_model

#File imports
from .hyperparams import model_name, model_args, reward_model_name, reward_model_args, checkpoint_location

#Download the base model and add its checkpoint
model = AutoModelForCausalLM.from_pretrained(model_name, **model_args)
checkpoint = torch.load(checkpoint_location)
model.load_state_dict(checkpoint["model_state"])

#Load in the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Download the reward model
reward_model = AutoModel.from_pretrained(reward_model_name, **reward_model_args).eval()