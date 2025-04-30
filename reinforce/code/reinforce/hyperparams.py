#Training loop hyperparameters
epochs = 50
val_interval = 1
smallest_val_loss = float("inf")
stop_patience = 4          #How many validation loops with no decrease in validation loss before the training stops
extra_id = 27
enter_id = 271

#Lr_scheduler hyperparameters
scheduler_patience = 3      #How many validation loops with no decrease in validation loss before the learning rate is multiplied by factor
scheduler_factor = .2       #The factor that the learning rate gets multiplied by when it is not improving

#Checkpoint hyperparameters
checkpoint_storage_location = "../../checkpoints/reinforce.pth"
graph_save_location = "../../outputs/graphs"

#Model generation args hyperparameters
generation_args = {
    "max_new_tokens" : 500
}


