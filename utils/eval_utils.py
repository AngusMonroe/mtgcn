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
    print(output.shape)
    print(labels.shape)
    k1 = 0
    k2 = 0
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            k1 += (output[i][j] == 1)
    print(k1)
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            k2 += (labels[i][j] == 1)
    print(k2)
    TP = ((output == 1) & (labels == 1)).sum()
    TN = ((output == 0) & (labels == 0)).sum()
    FP = ((output == 1) & (labels == 0)).sum()
    FN = ((output == 0) & (labels == 1)).sum()
    print(TP)
    print(TN)
    print(FP)
    print(FN)
    accuracy = accuracy_score(labels, output)
    f1_micro = f1_score(labels, output, average='micro')
    print(output)
    print(labels)
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

