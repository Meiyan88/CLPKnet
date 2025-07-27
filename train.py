import platform
import os
import json
from importlib import import_module as ImP_M
from Operation.Preparation import getNameOfExp, printTime, determineDevice
from Operation.GetDataset import getDataset, getDataLoader
from Operation.GetNetwork import getNetwork, printBNBiasFromLayers
from Operation.SaveLoadResume import *
from Operation.GenerateLog import generateLog, updateLog
from time import time
from Operation.Preparation import addArgs, getConfig, getTemp
#from context_premodel import ContextPredictionModel
from Compare_method import model_all

def inspect_model(model):
    param_count = 0
    for param_tensor_str in model.state_dict():
        tensor_size = model.state_dict()[param_tensor_str].size()
        print(f"{param_tensor_str} size {tensor_size} = {model.state_dict()[param_tensor_str].numel()} params")
        param_count += model.state_dict()[param_tensor_str].numel()

def mainProgramme(args, config, temp):
    for i in range(0, 5):
        temp = getTemp()
        # get the experiment time,save folder path, network save name
        exp_time, save_folder, network_name = getNameOfExp(args, config, i)
        train_data = os.path.join(config["train_path"], 'split{}'.format(i), 'train1.txt')
        val_data = os.path.join(config["val_path"], 'split{}'.format(i), 'test1.txt')
        test_data = config["independent_dataset"]

        # get dataset and dataload
        train_dataset, val_dataset, test_dataset = getDataset(args, config, train_data, val_data, test_data)
        train_dataset_loader, val_dataset_loader, test_dataset_loader = getDataLoader(
            train_dataset, val_dataset, test_dataset, args, config
        )

        # convert to device
        # determineDevice(args)
        # get network
        our = True
        if our:
            network = getNetwork(args, config, i)
            print('our model')
        else:
            id = 1
            print('current model is:',id)
            network = model_all.compare_model(id)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            network.to(device)

        # contextpre_network = ContextPredictionModel(in_channels=512)
        # context_predictor_weights_path = r'/home/huangmeiyan/wby1/GliomaRecurrence/model_weight/predictor.pth'
        # contextpre_network.load_state_dict(torch.load(context_predictor_weights_path,map_location='cuda:6'))
        #contextpre_network.eval()


        # save the initial network parameter
        saveInitNetwork(network, network_name, args)
        torch.set_num_threads(8)
        if args.is_train:

            # train
            # Generate Log Files
            log = generateLog(args, config, save_folder)

            # if not isinstance(network, list):
            #     resenc_para = network.parameters()
            # else:
            #     resenc_para = network[0].parameters()
            #
            # para = list(resenc_para) #+ list(contextpre_network.parameters())

            # Set Optimizer
            optimizer = torch.optim.SGD(
                network.parameters() if not isinstance(network, list) else network[0].parameters(),
                lr=config["base_lr"],
                momentum=0.9,
                weight_decay=0.0001,
            )
            # optimizer = torch.optim.Adam(
            #     network.parameters() if not isinstance(network, list) else network[0].parameters(),
            #     lr=config["base_lr"],
            #     weight_decay=0.0001,
            # )
            # print('current are adam')
            # optimizer = torch.optim.AdamW(
            #     network.parameters() if not isinstance(network, list) else network[0].parameters(),
            #     lr=config["base_lr"],
            #     weight_decay=0.0001,
            # )
            # set Lr_Scheduler
            lr_scheduler = None
            if config["lr_schedule"] == "OneCycleLR":
                lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=config["base_lr"],
                    steps_per_epoch=len(train_dataset_loader),
                    epochs=config["epochs"],
                    last_epoch=temp["scheduler_last_epoch"] - 1,
                )
            # torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #
            #     T_max=config["epochs"] * len(train_dataset_loader),  # 总迭代次数（steps）
            #     eta_min=1e-6,  # 最小学习率
            #     last_epoch=temp["scheduler_last_epoch"] - 1
            # )

            # begain to train
            for epoch in range(temp["current_epoch"], config["epochs"]):
                #print(torch.cuda.memory_summary())
                train_iter_paras = (optimizer, lr_scheduler, train_dataset_loader, epoch)
                conf_paras = (args, config)
                # train iteration
                train = ImP_M("Iteration.{}Iteration".format(args.exp_type)).train
                log_metrics = train(network, *train_iter_paras, *conf_paras, save_folder,train_dataset) #,contextpre_network

                # update some temporary value about epoch
                updateTempEpoch(temp, epoch, lr_scheduler)

                # val iteration
                with torch.no_grad():
                    val = ImP_M("Iteration.{}Iteration".format(args.exp_type)).val
                    log_metrics = val(network, val_dataset_loader, log_metrics, *conf_paras, save_folder)

                    # test iteration
                    test = ImP_M("Iteration.{}Iteration".format(args.exp_type)).test
                    log_metrics = test(network, test_dataset_loader, log_metrics, *conf_paras, save_folder)

                saveParaByMetric1(network, temp, log_metrics, save_folder, network_name, config)


                saveCheckpoint(network, optimizer, save_folder, args, config, temp)
                updateLog(log, log_metrics)
                print()
                # torch.cuda.empty_cache()

            # delete checkpoint
            os.remove(os.path.join(save_folder, "Checkpoint.pth"))
            print("Finitsh the Experiment, Delete the Checkpoint")

        else:
            # test
            pass

    return args, config

if __name__ == '__main__':

    if platform.system() == "Windows":
        print("Run in Windows system")
    elif platform.system() == "Linux":
        print("Run in Linux system")
        try:
            os.chdir("/public/huangmeiyan/wby/GliomaRecurrence/")
        except:
            print('wrong')
    print("The current WorkPlace: {}".format(os.getcwd()))

    args = addArgs()
    config = getConfig(args)
    print(config)
    print(config["log_metrics"]["Loss_Acc"][9])

    temp = getTemp()
    args, config = mainProgramme(args, config, temp)
    args.resume_exp = "None"