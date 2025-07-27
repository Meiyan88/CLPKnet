config = {}

# Initialize Configuration
config["imagenet_pretrain"] = False

# Network Configuration
config["network"] = "UnetResnet"  # Choose the Network
config["network_layer"] = 50  # the Number of layer of Network
config["img_ch"] = 4  # the Number of the Channel of Outputel of Input
config["output_ch"] = 2  # the Number of the Channel of Output

# Dataset Configuration
config["train_batch_size"] = 32  #  train Batch Size
config["val_batch_size"] = 1  # val Batch Size
config["test_batch_size"] = 1  # test Batch Size
config["datalist_path"] = ""  # the Name of Data List
config["seed"] = 30  # the Number of Data Random Seed
config["percent"] = 0  # the Percent of Data
config["pin_memory"] = False
config["num_workers"] = 0

# Training Configuration
config["epochs"] = 100  # Number of Train Epochs
config["base_lr"] = 0.01  # Base Learning Rate
config["lr_schedule"] = "OneCycleLR"
config["early_stop_dice"] = None  # Learning Rate Schedule
config["freeze_layers"] = None
config["save_folder"] = "Save"

# metrics and log config
config["log_metrics"] = {"Loss": ["Epoch", "Lr", "Train_Loss", "Val_Loss"]}
config["save_metric"] = {"key": "Loss", "order": "min", "index": 2}