from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score
import numpy as np

def acc_f1(output, labels, average='binary'):
    preds = output.max(1)[1].type_as(labels)
    if preds.is_cuda:
        preds = preds.cpu()
        labels = labels.cpu()
    accuracy = accuracy_score(preds, labels)
    f1 = f1_score(preds, labels, average=average)
    return accuracy, f1


def acc_f1_auc(output, labels, average='binary'):
    accuracy = accuracy_score(output, labels)
    f1_micro = f1_score(output, labels, average='micro')
    f1_macro = f1_score(output, labels, average='macro')
    auc_micro = roc_auc_score(np.array(output), np.array(labels), average='micro')
    auc_macro = roc_auc_score(np.array(output), np.array(labels), average='macro')
    return accuracy, f1_micro, f1_macro, auc_micro, auc_macro

