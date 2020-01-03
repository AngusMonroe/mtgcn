from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
import numpy as np
import torch

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average=average)
    return accuracy, f1


def acc_f1_auc(output, labels, n_classes):
    output = torch.sigmoid(output)
    output = (output > 0.5).long()
    TP = (output == 1) & (labels == 1).sum()
    FP = (output == 1) & (labels == 0).sum()
    TN = (output == 0) & (labels == 0).sum()
    FN = (output == 0) & (labels == 1).sum()
    # accuracy = accuracy_score(labels, output)
    accuracy = (TP+TN) / (TP+TN+FP+FN)
    # f1_micro = f1_score(labels, output, average='micro')
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_micro = 2*precision*recall / (precision+recall)
    f1_macro = f1_score(labels, output, average='macro')
    labels = np.array(labels)
    output = np.array(output)
    fpr, tpr, _ = roc_curve(labels.ravel(), output.ravel())
    auc_micro = auc(fpr, tpr)
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(labels[:, i], output[:, i])
        aucs.append(auc(fpr, tpr))
    auc_macro = np.mean(aucs)
    return accuracy, f1_micro, f1_macro, auc_micro, auc_macro

