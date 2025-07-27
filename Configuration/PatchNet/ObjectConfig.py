from Configuration.ObjectConfig import config

# Network Configuration
config["network"] = "Resnet"
config["img_ch"] = 4
config["network_layer"] = 50
config["output_ch"] = [224, 2]

# Dataset Configuration
config["recons"] = ["TCIA"]
config["datalist_dir"] = "DataList/TCIA"
config["trainset"] = "train"
config["valset"] = "test"
config["testset"] = "test"
config["selection_strategy"] = "all_in"  # ['all_in', 'sample']

# Dataloader Configuration
config["train_batch_size"] = 1
config["val_batch_size"] = 1
config["test_batch_size"] = 1
config["pin_memory"] = False
config["num_workers"] = 0

# Data processing config
config["roi_h"] = 64  # -1 means there not roi   # padding when encounter boundary(x)
config["roi_w"] = 64
config["introduce_normal"] = False
config["is_only_mask_region"] = False

# Data Augmentation config
config["is_aug"] = True
config["aug_Rotate"] = 0.25
config["aug_GaussianNoise"] = 0.25
config["aug_Gamma"] = 0.25
config["aug_Flip_x"] = 0.25
config["aug_Flip_y"] = 0.25

# Loss config
config["fix_index_rep_recon_fea"] = None
config["fix_index_rep_recon"] = None

# Training Configuration
config["save_folder"] = "Save"
config["base_lr"] = 0.005
config["epochs"] = 100
config["is_classification_task"] = False
config["is_segmentation_task"] = False
config["seg_loss_rate"] = 0.5
config["tf_loss_rate"] = 0

# iteration configuration
config["terminal_KF"] = 5
config["num_KF"] = 5

# test configuration
config["is_init_parameter"] = False

# Debug Special config
config["show_patient_id"] = []
config["is_max_mask_slice"] = False
config["show"] = [False, False, False, False, False]

# metrics and log config
config["log_metrics"] = {}
config["log_metrics"]["Loss_Acc"] = ["Epoch", "Lr", "Train_Loss", "Train_Acc", "Val_Loss", "Val_Acc"]
config["log_metrics"]["Loss_Detail"] = [
    "Epoch",
    "Train_Class_Loss",
    "Train_Segme_Loss",
    "Train_TF_Loss",
    "Val_Class_Loss",
    "Val_Segme_Loss",
    "Val_TF_Loss"
]
config["save_metric"] = {"key": "Loss_Acc", "order": "max", "index": 5}