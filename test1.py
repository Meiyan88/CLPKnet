import Operation.Utils as U
import torch
import numpy as np
import os
import pandas as pd
from Operation.Preparation import addArgs, getConfig, getTemp
########################### ########################### ########################### ###########################
# Network
from Network.MSFEnetwork import MSFEnet
########################### ########################### ########################### ###########################
from Dataset.MSFEDataset import MSFEDataset
from Compare_method import model_all
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_curve,
)
from sklearn import metrics
# import matplotlib.pyplot as plt

def get_metrics(pred_scores, gt_labels, threshold, is_print=False):
    if pred_scores.shape[0] == 0 or gt_labels.shape[0] == 0:
        return 0, 0, 0, 0, 0, None

    pred_labels = (pred_scores > threshold).astype(np.uint8)

    if is_print:
        pred_print = pred_labels.reshape(-1, pred_labels.shape[0])
        print(pred_print)

    acc = accuracy_score(gt_labels, pred_labels)

    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    #cm = confusion_matrix(gt_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_true=gt_labels, y_pred=pred_labels).ravel()
    specificity = tn / (tn + fp + 1e-6)
    sensitivity = tp / (tp + fn + 1e-6)

    return (
        round(float(auc), 4),          # AUC
        round(float(acc), 4),          # 准确率
        round(float(specificity), 4),  # 特异性
        round(float(sensitivity), 4)   # 灵敏度
    )
def unfold(x):
    patch_size = 32
    num_patches_h = 96 // 32  # 3
    num_patches_w = 96 // 32  # 3

    # 使用 unfold 切分空间维度
    patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # (32, 4, 3, 3, 32, 32)

    # 合并批次和 patch 维度
    patches = patches.permute(0, 2, 3, 1, 4, 5)  # (32, 3, 3, 4, 32, 32)
    patches = patches.reshape(-1, 4, 32, 32)  # (32*9=288, 4, 32, 32)

    return patches
if __name__ == '__main__':
    from Configuration.PatchNet.MSFE_Class_NFtrainZJtest import config
    torch.set_num_threads(8)
    args = addArgs()

    model_path = r'D:\GliomaRecurrence\Save\CLPKnet'
    model_list = ['KF0\epoch_38.pth',
                  'KF1\epoch_32.pth',
                  'KF2\epoch_3.pth',
                  'KF3\epoch_38.pth',
                  'KF4\epoch_12.pth']
    test_data_path = config["independent_dataset"]
    #savepath = r'C:\Users\61453\Desktop\GliomaRecurrence\Save\Multi_center\img_result'
    savepath = model_path

    dataset = MSFEDataset(args, config, root=test_data_path, train_val_test='test')
    test_dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    sum_batchsize = len(test_dataset_loader)

    FiveFold_list_scores = [[], [], [], [], []]
    FiveFold_list_labels = [[], [], [], [], []]
    pred_result_one = []
    pred_result_all = []
    pred_result_both = []
    
    our = True
    for i, model in enumerate(model_list):
        print(i)
        with torch.no_grad():
            if our:
                network = MSFEnet(config=config)
                network.load_state_dict(torch.load(os.path.join(model_path, model), map_location='cpu'))
            else:
                id = 6
                print('current model is:', id)
                network = model_all.compare_model(id)
                network.load_state_dict(torch.load(os.path.join(model_path, model), map_location='cpu'))

            network.cuda(0)
            network.eval()

            for images, labels,_ in test_dataset_loader:
                labels = labels.view(-1, 9)
                labels = labels.view(-1)

                images = images.cuda(0)
                labels = labels.cuda(0)

                if our:
                    _,outputs_c = network(images)
                else:
                    images1 = unfold(images)
                    outputs_c = network(images1,train=False)

                labels = labels.type(torch.float)
                labels = labels.reshape(labels.shape[0], -1)
                scores = U.toNumpy(outputs_c, is_squeeze=False)  # (B,C)
                labels = U.toNumpy(labels, np.int64, is_squeeze=False)  # (B,C)

                FiveFold_list_scores[i].append(scores)
                FiveFold_list_labels[i].append(labels)


    scores_mean_5F = []
    for k in range(len(FiveFold_list_labels[0])):
        a = FiveFold_list_scores[0][k] + FiveFold_list_scores[1][k] + FiveFold_list_scores[2][k] + FiveFold_list_scores[3][k] + FiveFold_list_scores[4][k]
        scores_mean_5F.append(a)

    list_scores = np.concatenate(scores_mean_5F, axis=0)  # (N,C)
    list_labels = np.concatenate(FiveFold_list_labels[0], axis=0)  # (N)


    # 1. 存储每个模型的指标
    model_metrics = []

    for fold_idx in range(len(model_list)):
        # 获取当前模型的预测分数和标签
        fold_scores = np.concatenate(FiveFold_list_scores[fold_idx], axis=0)
        fold_labels = np.concatenate(FiveFold_list_labels[fold_idx], axis=0)

        # 计算该模型的指标（以threshold=0为例）
        auc, acc, specificity, sensitivity = get_metrics(
            pred_scores=fold_scores,
            gt_labels=fold_labels,
            threshold=0,
            is_print=False
        )
        print(auc,acc,specificity,sensitivity)

        # 记录指标
        model_metrics.append({
            'Fold': fold_idx,
            'AUC': auc,
            'Accuracy': acc,
            'Specificity': specificity,
            'Sensitivity': sensitivity
        })

    # 2. 计算均值和标准差
    metrics_df = pd.DataFrame(model_metrics)
    mean_metrics = metrics_df.mean(axis=0).rename('Mean')
    std_metrics = metrics_df.std(axis=0).rename('Std')

    # 3. 合并统计结果
    stats_df = pd.concat([mean_metrics, std_metrics], axis=1)
    stats_df = stats_df.drop(['Fold'])  # 排除非数值列

    # 4. 保存到文件（与集成模型结果分开保存）
    stats_df.to_csv(os.path.join(savepath, 'independent_model_std4.csv'))


