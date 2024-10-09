import sys
import os
import pandas as pd
import numpy as np
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
from networkx.readwrite import json_graph
import torch
import torch.nn as nn
import copy
import seaborn
from sklearn.metrics import calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from AOP_NN import AOP_NN
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.metrics import auc, roc_auc_score, roc_curve, precision_recall_curve, accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef, precision_score, recall_score

def Predict(seed, num_loop, label_name, feature, samples, KF):

    label_test = samples.loc[:, label_name]
    labels = torch.tensor(np.array(label_test)).long()

    data = torch.tensor(np.array(samples.iloc[:, 3:])).float()
    cuda_labels = labels
    test = np.array(cuda_labels, dtype=np.float32)

    test_predict_soft2_list = []
    for ii in range(num_loop):
        model = torch.load('Model/'+str(KF)+'/'+str(seed)+'_'+str(feature)+'_'+str(label_name) + '_' + str(ii) + '.pth')

        model.eval()
        aux_out_map, term_NN_out_map = model(data)
        test_predict = aux_out_map['final'].data
        test_predict_soft1 = torch.softmax(test_predict, dim=-1)
        indices = torch.tensor([1])
        test_predict_soft2 = torch.index_select(test_predict_soft1, -1, indices)
        test_predict_soft2_list.append(np.array(test_predict_soft2, dtype=np.float32))

    pre_p = np.mean(np.array(test_predict_soft2_list), axis=0)
    pre = np.where(pre_p > 0.5, 1, 0)
    precision1, recall1, thresholds = precision_recall_curve(test, pre)

    dic_test = [roc_auc_score(test, pre_p), accuracy_score(test, pre), auc(recall1, precision1),
                         balanced_accuracy_score(test, pre), f1_score(test, pre), precision_score(test, pre),
                         recall_score(test, pre), matthews_corrcoef(test, pre)]

    results = pd.DataFrame(dic_test, index=['AUROC', 'Accuracy', 'AUPR', 'BACC', 'F1 score', 'Precision', 'Recall', 'MCC']).T
    results.loc['VNN', :] = [np.mean(results.loc[:, i]) for i in results.columns]
    # results.to_csv('Result/TestResults_KF_'+str(KF)+'_'+str(feature)+'_'+str(label_name)+'.csv')
    print('-------------------------------------------------------------test results: ', results.loc['VNN', :])

    return results


