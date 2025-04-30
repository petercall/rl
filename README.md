I have included a requirements.txt file which gives the libraries I have downloaded for a Python virtual environment.
You can install them via: pip install -r requirements.txt

As I put in my slides, I trained several models. One is a supervised-fine-tuned model, which is contained in the sft folder.
Another is a LoRA tuned problem, which is in the lora folder.
The third is the reward model, which is in the reward_model folder.
The last is the REINFORCE model, which is in the reinforce model.

You can train the model yourself in any of these folders by going into the folder: code/NAME 
where you replace NAME with either sft, lora, reward_model, or reinforce.
You then run the python file as: python train_NAME.py where NAME is either sft, lora, reward_model, or reinforce.

The trained checkpoint that I received for the models is in the folder: checkpoints
inside of each folder.

Note: When you train the model, it will download a model from hugging face that is several gigabites, which it uses as the base model. Just so you are aware of that.
