import torch
from Operation.Utils import AverageMeter, InterAndUnion
import numpy as np
from PIL import Image
import os
import torch
from torch import nn
import torch.nn.functional as F
from Operation.Utils import DFAdd2CSV, getShapes, merge, split, toNumpy
from Operation.Metric import calculateAUCAndPRF1
import pandas as pd
import itertools
import pickle
from sklearn.metrics import precision_recall_fscore_support

import matplotlib.pyplot as plt
import Operation.Utils as U
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    roc_curve,
)
import random
from scipy import ndimage
from sklearn import metrics

from sklearn.linear_model import LogisticRegression





def smooth(label, smoothing=0.0):
    assert 0 <= smoothing < 1
    return label * (1.0 - smoothing) + 0.5 * smoothing


def round_(number, id=3):
    return round(number, id)


def get_metrics(pred_scores, gt_labels, is_print=False):
    if pred_scores.shape[0] == 0 or gt_labels.shape[0] == 0:
        return 0, 0, 0, 0, 0, None

    pred_labels = (pred_scores > 0).astype(np.uint8)

    if is_print:
        pred_print = pred_labels.reshape(-1, pred_labels.shape[0])
        print(pred_print)

    acc = accuracy_score(gt_labels, pred_labels)

    p, r, f1, _ = precision_recall_fscore_support(
        gt_labels, pred_labels, pos_label=1, average="binary", zero_division=0
    )

    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    cm = confusion_matrix(gt_labels, pred_labels)
    tn, fp, fn, tp = confusion_matrix(y_true=gt_labels, y_pred=pred_labels).ravel()
    specificity = tn / (tn + fp + 1e-6)
    sensitivity = tp / (tp + fn + 1e-6)

    return (float(round_(auc)), float(round_(acc)), float(round_(p)), float(round_(r)), float(round_(f1)), cm,
            float(round_(specificity)), float(round_(sensitivity)))


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


our = True
HCP = False
########################################################################################################
# # 2D train
def train(network, optimizer, lr_scheduler, train_dataset_loader, epoch, args, config, save_folder,
          train_dataset):  # ,contextpre_network

    loss_fn_bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(data=[6.0], dtype=torch.float32).cuda(0)  # config["BCE_pos_weight"]
    )

    Sigmoid = torch.nn.Sigmoid()

    # train loss logger
    train_loss = AverageMeter()
    class_loss = AverageMeter()
    list_scores = []
    list_labels = []

    pred_result_one = []
    pred_result_all = []
    pred_result_both = []
    topk = config["topK_list"]

    # freeze running variable
    if config["freeze_layers"] is not None:
        for (name, layers) in network.named_children():
            if name in config["freeze_layers"]:
                # print('freeze the {}'.format(name))
                layers.eval()

    network.train()

    sum_batchsize = len(train_dataset_loader)


    # begin to iterate
    for i, (images, labels,weight) in enumerate(train_dataset_loader):
        # get the learning rate
        lr = optimizer.param_groups[0]["lr"]
        batchsize = config["train_batch_size"]
        current_batchsize = labels.shape[0]

        labels = labels.view(-1, 9)
        labels = labels.view(-1)

        images = images.cuda(0)
        labels = labels.cuda(0)
        labels = smooth(labels,smoothing=0.1)

        # forward
        weight = weight.cuda(0)
        weight = weight.unsqueeze(1)
        weight = nn.Parameter(weight)
        if our:
            output_mid, outputs_c = network(images)  # )
        else:
            if HCP:
                outputs_c = network(images)
            else:
                images = unfold(images)
                outputs_c = network(images)
        # calculate loss value
        loss = 0
        class_loss_ = 0
        #################### calculate classfication loss ####################
        labels = labels.type(torch.float)
        labels = labels.reshape(labels.shape[0], -1)

        if outputs_c is not None:
            class_loss_ = loss_fn_bce(outputs_c, labels)
        outputs_c_sigmoid = Sigmoid(outputs_c)
        prior = True

        if our and prior:
            weight = weight.cuda(0)
            weight = nn.Parameter(weight)
            weight = weight.type(torch.float)
            weight = weight.reshape(labels.shape[0], -1)
            not_on_white = weight * outputs_c
            loss_white = (not_on_white).mean()

        scores_sigmoid = U.toNumpy(outputs_c_sigmoid, is_squeeze=False)
        scores = U.toNumpy(outputs_c, is_squeeze=False)  # (B,C)
        labels = U.toNumpy(torch.round(labels), np.int64, is_squeeze=False)  # (B,C)
        list_scores.append(scores)  # NB * (B,C)
        list_labels.append(labels)  # NB * (B)

        for sample in range(0, current_batchsize):
            topk_index_list = []
            sub_set_data = scores_sigmoid[sample * 9: (sample + 1) * 9].flatten()
            sub_set_label = labels[sample * 9: (sample + 1) * 9].flatten()
            positive_prob_sorted = sorted(sub_set_data, reverse=True)

            for m in range(topk):
                rate = positive_prob_sorted[m]
                rate_index = np.where(sub_set_data == rate)
                if len(rate_index[0]) != 1:
                    for index_sub in rate_index[0]:
                        if index_sub in topk_index_list:
                            pass
                        else:
                            topk_index_list.append(index_sub)
                            break
                else:
                    topk_index_list.append(rate_index)
            label_index = np.where(sub_set_label == 1)

            marker_one = 0
            marker_all = 1
            for l in label_index[0]:
                if l in topk_index_list:
                    marker_one = 1
                if l not in topk_index_list:
                    marker_all = 0
            pred_result_one.append(marker_one)
            pred_result_all.append(marker_all)

            marker_both = 1
            for b0th in topk_index_list:
                if b0th not in label_index[0]:
                    marker_both = 0
            pred_result_both.append(marker_both)

        # loss_aux = contextpredict(img256.cuda(2),network,contextpre_network.cuda(2),dataset_train=train_dataset)
        if prior and our:
            loss = loss + class_loss_ * config["class_loss_rate"] + loss_white * 0.1  # + loss_aux * 0.01#+ +  loss_white * 0.05
        else:
            loss = loss + class_loss_ * config["class_loss_rate"]

        # log the loss value
        train_loss.update(loss.item())
        class_loss.update(class_loss_.item() if class_loss_ != 0 else 0)
        # segme_loss.update(segme_loss_.item() if segme_loss_ != 0 else 0)
        # train_dice.update(Dice_value[0][1] if Dice_value[0][1] != 0 else 0)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        if (i + 1) % int(np.sqrt(sum_batchsize)) == 0 or i + 1 == sum_batchsize:
            print("BatchSize: {}/{} Train Loss: {:.6f} lr: {:.6f}".format(i + 1, sum_batchsize, train_loss.avg, lr))
        # break

    log_metrics = {}
    log_metrics["Loss_Acc"] = [epoch + 1, lr, train_loss.avg]
    log_metrics["Loss_Detail"] = [epoch + 1, class_loss.avg]

    list_scores = np.concatenate(list_scores, axis=0)  # (N,C)
    list_labels = np.concatenate(list_labels, axis=0)  # (N)

    auc, acc, p, r, f1, cm, spe, sen = get_metrics(pred_scores=list_scores,
                                                   gt_labels=list_labels,
                                                   is_print=False)

    log_metrics["Loss_Acc"].append(acc)
    log_metrics["Loss_Acc"].append(auc)
    log_metrics["Loss_Acc"].append(spe)
    log_metrics["Loss_Acc"].append(sen)


    train_data_savepath = os.path.join(save_folder, "train_data_record")
    if not os.path.exists(train_data_savepath):
        os.makedirs(train_data_savepath)
    np.savez(os.path.join(train_data_savepath, "train_data_{}.npz".format(str(epoch + 1))),
             train_data=list_scores, train_label=list_labels)

    return log_metrics


############################################################################################################
# 2D val
def val(network, val_dataset_loader, log_metrics, args, config, save_folder):
    # val

    network.eval()

    # get the number of iterations each epoch
    sum_batchsize = len(val_dataset_loader)

    # loss function
    loss_fn_bce = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(data=config["BCE_pos_weight"], dtype=torch.float32).cuda(0)
    )

    Sigmoid = torch.nn.Sigmoid()

    # train loss logger
    val_loss = AverageMeter()
    class_loss = AverageMeter()
    list_scores = []
    list_labels = []

    pred_result_one = []
    pred_result_all = []
    pred_result_both = []
    topk = config["topK_list"]

    # freeze running variable
    if config["freeze_layers"] is not None:
        for (name, layers) in network.named_children():
            if name in config["freeze_layers"]:

                layers.eval()

    # begin to iterate
    for i, (images, labels) in enumerate(val_dataset_loader):
        # get the learning rate
        batchsize = config["val_batch_size"]

        current_batchsize = labels.shape[0]

        labels = labels.view(-1, 9)

        labels = labels.view(-1)

        images = images.cuda(0)
        labels = labels.cuda(0)

        # weight = weight.cuda(0)
        # weight = nn.Parameter(weight)
        # weight = weight.unsqueeze(1)

        # forward
        # weight = weight.cuda(0)


        if our:
            output_mid, outputs_c = network(images)  # )
        else:
            if HCP:
                outputs_c = network(images)
            else:
                images = unfold(images)
                outputs_c = network(images)

        # calculate loss value
        loss = 0
        class_loss_ = 0
        #################### calculate classfication loss ####################
        labels = labels.type(torch.float)
        labels = labels.reshape(labels.shape[0], -1)

        if outputs_c is not None:
            class_loss_ = loss_fn_bce(outputs_c, labels)
            # class_loss_ = py_softmax_focal_loss(outputs_c, labels)
        outputs_c_sigmoid = Sigmoid(outputs_c)
        scores_sigmoid = U.toNumpy(outputs_c_sigmoid, is_squeeze=False)
        scores = U.toNumpy(outputs_c, is_squeeze=False)  # (B,C)
        labels = U.toNumpy(labels, np.int64, is_squeeze=False)  # (B,C)
        list_scores.append(scores)  # NB * (B,C)
        list_labels.append(labels)  # NB * (B)

        for sample in range(0, current_batchsize):
            topk_index_list = []
            sub_set_data = scores_sigmoid[sample * 9: (sample + 1) * 9].flatten()
            sub_set_label = labels[sample * 9: (sample + 1) * 9].flatten()
            positive_prob_sorted = sorted(sub_set_data, reverse=True)

            for m in range(topk):
                rate = positive_prob_sorted[m]
                rate_index = np.where(sub_set_data == rate)
                if len(rate_index[0]) != 1:
                    for index_sub in rate_index[0]:
                        if index_sub in topk_index_list:
                            pass
                        else:
                            topk_index_list.append(index_sub)
                            break
                else:
                    topk_index_list.append(rate_index)
            label_index = np.where(sub_set_label == 1)

            marker_one = 0
            marker_all = 1
            for l in label_index[0]:
                if l in topk_index_list:
                    marker_one = 1
                if l not in topk_index_list:
                    marker_all = 0
            pred_result_one.append(marker_one)
            pred_result_all.append(marker_all)

            marker_both = 1
            for b0th in topk_index_list:
                if b0th not in label_index[0]:
                    marker_both = 0
            pred_result_both.append(marker_both)
        # break
        loss = loss + class_loss_ * config["class_loss_rate"]

        # log the loss value
        val_loss.update(loss.item())
        class_loss.update(class_loss_.item() if class_loss_ != 0 else 0)

    log_metrics["Loss_Acc"] += [val_loss.avg]
    log_metrics["Loss_Detail"] += [class_loss.avg]

    list_scores = np.concatenate(list_scores, axis=0)  # (N,C)
    list_labels = np.concatenate(list_labels, axis=0)  # (N)

    # pred, gt = list_scores, list_labels
    # pred = pred[gt != -1]
    # gt = gt[gt != -1]
    auc, acc, p, r, f1, cm, spe, sen = get_metrics(pred_scores=list_scores,
                                                   gt_labels=list_labels,
                                                   is_print=False)
    # print(cm.tolist())
    log_metrics["Loss_Acc"].append(acc)
    log_metrics["Loss_Acc"].append(auc)
    log_metrics["Loss_Acc"].append(spe)
    log_metrics["Loss_Acc"].append(sen)

    val_data_savepath = os.path.join(save_folder, "val_data_record")
    if not os.path.exists(val_data_savepath):
        os.makedirs(val_data_savepath)
    np.savez(os.path.join(val_data_savepath, "val_data_{}.npz".format(log_metrics["Loss_Acc"][0])),
             val_data=list_scores, val_label=list_labels)

    return log_metrics


############################################################################################################
# 2D test
def test(network, test_dataset_loader, log_metrics, args, config, save_folder):
    # val
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3, 4, 0"
    network.eval()

    # get the number of iterations each epoch
    sum_batchsize = len(test_dataset_loader)

    Sigmoid = torch.nn.Sigmoid()

    # train loss logger
    list_scores = []
    list_labels = []

    pred_result_one = []
    pred_result_all = []
    pred_result_both = []
    topk = config["topK_list"]

    # freeze running variable
    if config["freeze_layers"] is not None:
        for (name, layers) in network.named_children():
            if name in config["freeze_layers"]:
                # print('freeze the {}'.format(name))
                layers.eval()

    # begin to iterate
    for i, (images, labels,_) in enumerate(test_dataset_loader):
        # get the learning rate
        batchsize = config["test_batch_size"]
        current_batchsize = labels.shape[0]

        labels = labels.view(-1, 9)
        labels = labels.view(-1)

        images = images.cuda(0)
        labels = labels.cuda(0)

        # weight = weight.cuda(0)
        # weight = nn.Parameter(weight)

        # forward
        # weight = weight.cuda(0)
        # weight = weight.unsqueeze(1)
        # #print(weight.shape)
        # weight = nn.Parameter(weight)

        if our:
            output_mid, outputs_c = network(images)  #)
        else:
            if HCP:
                outputs_c = network(images)
            else:
                images = unfold(images)
                outputs_c = network(images)


        labels = labels.type(torch.float)
        labels = labels.reshape(labels.shape[0], -1)

        outputs_c_sigmoid = Sigmoid(outputs_c)
        scores_sigmoid = U.toNumpy(outputs_c_sigmoid, is_squeeze=False)
        scores = U.toNumpy(outputs_c, is_squeeze=False)  # (B,C)
        labels = U.toNumpy(labels, np.int64, is_squeeze=False)  # (B,C)
        list_scores.append(scores)  # NB * (B,C)
        list_labels.append(labels)  # NB * (B)

        for sample in range(0, current_batchsize):
            topk_index_list = []
            sub_set_data = scores_sigmoid[sample * 9: (sample + 1) * 9].flatten()
            sub_set_label = labels[sample * 9: (sample + 1) * 9].flatten()
            positive_prob_sorted = sorted(sub_set_data, reverse=True)

            for m in range(topk):
                rate = positive_prob_sorted[m]
                rate_index = np.where(sub_set_data == rate)
                if len(rate_index[0]) != 1:
                    for index_sub in rate_index[0]:
                        if index_sub in topk_index_list:
                            pass
                        else:
                            topk_index_list.append(index_sub)
                            break
                else:
                    topk_index_list.append(rate_index)
            label_index = np.where(sub_set_label == 1)

            marker_one = 0
            marker_all = 1
            for l in label_index[0]:
                if l in topk_index_list:
                    marker_one = 1
                if l not in topk_index_list:
                    marker_all = 0
            pred_result_one.append(marker_one)
            pred_result_all.append(marker_all)

            marker_both = 1
            for b0th in topk_index_list:
                if b0th not in label_index[0]:
                    marker_both = 0
            pred_result_both.append(marker_both)
        # break

    list_scores = np.concatenate(list_scores, axis=0)  # (N,C)
    list_labels = np.concatenate(list_labels, axis=0)  # (N)

    # pred, gt = list_scores, list_labels
    # pred = pred[gt != -1]
    # gt = gt[gt != -1]
    auc, acc, p, r, f1, cm, spe, sen = get_metrics(pred_scores=list_scores,
                                                   gt_labels=list_labels,
                                                   is_print=False)
    # print(cm.tolist())
    log_metrics["Loss_Acc"].append(acc)
    log_metrics["Loss_Acc"].append(auc)
    log_metrics["Loss_Acc"].append(spe)
    log_metrics["Loss_Acc"].append(sen)
    # log_metrics["Loss_Acc"].append(sum(pred_result_one) / len(pred_result_one))
    # log_metrics["Loss_Acc"].append(sum(pred_result_all) / len(pred_result_all))
    # log_metrics["Loss_Acc"].append(sum(pred_result_both) / len(pred_result_both))

    val_data_savepath = os.path.join(save_folder, "test_data_record")
    if not os.path.exists(val_data_savepath):
        os.makedirs(val_data_savepath)
    np.savez(os.path.join(val_data_savepath, "test_data_{}.npz".format(log_metrics["Loss_Acc"][0])),
             test_data=list_scores, test_label=list_labels)

    print("epoch:{} lr:{:.4f} trian_loss:{:.4f} \n\
    train_acc:{:.4f} train_auc:{:.4f} train_spe:{:.4f} train_sen:{:.4f}\n\
    val_Loss:{:.4f}  val_acc:{:.4f} val_auc:{:.4f} val_spe:{:.4f} val_sen:{:.4f} \n\
    test_acc:{:.4f} test_auc:{:.4f} test_spe:{:.4f} test_sen:{:.4f} "
          .format(*log_metrics["Loss_Acc"]))
    print(
        "epoch:{} train_class_loss:{:.4f} val_class_loss:{:.4f}".format(
            *log_metrics["Loss_Detail"]
        )
    )
    return log_metrics