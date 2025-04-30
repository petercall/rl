#Training loop hyperparameters
epochs = 50
val_interval = 1
smallest_val_loss = float("inf")
stop_patience = 5          #How many validation loops with no decrease in validation loss before the training stops

#Lr_scheduler hyperparameters
scheduler_patience = 3      #How many validation loops with no decrease in validation loss before the learning rate is multiplied by factor
scheduler_factor = .2       #The factor that the learning rate gets multiplied by when it is not improving

#Checkpoint hyperparameters
checkpoint_storage_location = "../../checkpoints/sft.pt"
graph_save_location = "../../outputs/graphs"



