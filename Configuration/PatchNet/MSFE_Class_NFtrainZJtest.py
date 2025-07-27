from Configuration.PatchNet.ObjectConfig import config

###########train path & val path##########
config["train_path"] = r"G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\datasplit\FLAIR_T1_T1C_T2"
config["val_path"] = r"G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\datasplit\FLAIR_T1_T1C_T2"
config["independent_dataset"] = r"G:\GliomaRecurrence\Datalist\OnlyPosOperation_First\datasplit\ZhuJiang_sum.txt"
config["save_folder"] = r"G:\GliomaRecurrence\Save"

config["modalities"] = ["MRI, FLAIR, T2, T1, T1C"]
config["orders_img"] = "C H W"
config["resolution"] = []
config["crop_shape"] = []
config["resize_shape"] = [128, 128]
config["only_mask_region"] = False
config["cat_mask"] = False
config["val_test_crop_shape"] = None
config["pred_target"] = ["Recurr"]
config["data_preprocess_detail"] = ""

# config["number_Kfold"] = 4
########### Network Configuration##########
config["network"] = "Resnet18_Transformer"
config["network_layer"] = "18"
config["output_ch"] = [1]
config["freeze_layers"] = None
config["topK_list"] = 3

########### Learning Rate Schedule Config##########
config["lr_schedule"] = "OneCycleLR"
config["weight_decay"] = 0.001
config["base_lr"] = 0.005
config["StepLR_step_size"] = 50
config["StepLR_gamma"] = 0.1
config["MultiStepLR_milestones"] = [30, 80, 150]
config["MultiStepLR_gamma"] = 0.1

########### Dataset Configuration##########
config["train_batch_size"] = 32
config["val_batch_size"] = 1
config["test_batch_size"] = 1
config["recons"] = [""]

########### Data Augmentation config##########
config["is_aug"] = True
config["aug_p"] = 0.5
config["aug_Rotate"] = 0.20
config["aug_rotate_set"] = 15
config["aug_Flip_x"] = 0.20
config["aug_Flip_y"] = 0.20
config["aug_Flip_z"] = 0.20
config["aug_translation"] = 0.20

config["aug_intensity_gamma"] = 1
config["aug_intensity_gamma_set"] = (0.8, 1.2)

config["aug_intensity_shift"] = 0
config["aug_intensity_shift_set"] = (-0.1, 0.1)
config["aug_intensity_scale"] = 0
config["aug_intensity_scale_set"] = (0.9, 1.1)

config["aug_gaussian_noise"] = 1
config["aug_gaussian_noise_set"] = 0.025
config["aug_gaussian_smooth"] = 1
config["aug_gaussian_smooth_set"] = 0.20

########### Training Configuration##########
config["is_classification_task"] = True
config["is_segmentation_task"] = False

# Debug Special config
config["show"] = [False, False, False, False, False]  # slice, roi, pre-processing, by mask, aug

###########loss rate##########
config["imagenet_pretrain"] = False
config["seg_loss_rate"] = 0
config["class_loss_rate"] = 1

########## Loss Configuration ##########
config["loss_fn"] = "BCE"
config["BCE_pos_weight"] = [7.0]
config["weight_balancing"] = True

##########metrics and log config##########
config["log_metrics"] = {}
config["log_metrics"]["Loss_Detail"] = ["Epoch", "Train_class_Loss", "Val_class_loss"]
config["log_metrics"]["Loss_Acc"] = [
    "Epoch",
    "lr",
    "train_loss",
    "train_acc",
    "train_auc",
    "train_spe",
    "train_sen",
    "val_loss",
    "val_acc",
    "val_auc",
    "val_spe",
    "val_sen",
    "test_acc",
    "test_auc",
    "test_spe",
    "test_sen",
]
config["save_metric"] = {"key": "Loss_Acc", "order": "max", "index": 12}