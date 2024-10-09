# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import time
import networkx as nx
import networkx.algorithms.dag as nxadag
import networkx.algorithms.dag as nxaldag
import torch
import torch.nn as nn
import torch.utils.data as du
from AOP_NN import AOP_NN
from early_stopping import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# initialize weight param
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.kaiming_uniform_(m.weight)
        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# define train_model function
def train(seed, a, batch_size, loop, epoch, label_name, feature, learning_rate, dG, input_dim, num_hiddens_ratio, traindata, validdata, KF):

    model = AOP_NN(dG, input_dim, num_hiddens_ratio)
    model.apply(weight_init)
    # total_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-5, weight_decay=1e-4)
    optimizer.zero_grad()
    early_stopping = EarlyStopping(seed, feature, label_name, loop, 'Model/' + str(KF))

    valid_loader = DataLoader(dataset=validdata, batch_size=batch_size, shuffle=True)

    acc_v_list = []
    loss_v_list = []
    loss_v = 0.0
    for e in range(epoch):
        model.train()
        optimizer.zero_grad()

        train_loader = DataLoader(dataset=traindata, batch_size=batch_size, shuffle=True)
        batches = list(train_loader)
        if len(batches[-1][0]) < batch_size / 2:
            batches = batches[:-1]

        total_acc_train = 0.0
        for m, (inputdata, labels) in enumerate(batches):
            cuda_features = inputdata
            cuda_labels = labels

            aux_out_map, _ = model(cuda_features)
            train_predict = aux_out_map['final'].data

            total_loss = 0.0
            final_loss = 0.0
            branch_loss = 0.0
            for name, output in aux_out_map.items():

                if name == 'final':
                    final_loss += nn.CrossEntropyLoss()(output, cuda_labels).item()
                    total_loss += nn.CrossEntropyLoss()(output, cuda_labels)
                else:
                    total_loss += a * nn.CrossEntropyLoss()(output, cuda_labels)
                    branch_loss += a * nn.CrossEntropyLoss()(output, cuda_labels).item()

            total_loss.backward()
            optimizer.step()

            # train_acc compute
            train_predict_soft = torch.softmax(train_predict, dim=-1)
            train_predict_soft = torch.argmax(train_predict_soft, dim=-1)
            num_correct = (cuda_labels == train_predict_soft).sum().item()
            train_acc = num_correct / len(train_predict_soft)
            total_acc_train += train_acc

        acc_v_all = 0.0
        loss_v_all = 0.0
        model.eval()
        for m, (inputdata, labels) in enumerate(valid_loader):
            cuda_features = inputdata
            cuda_labels = labels

            aux_out_map, _ = model(cuda_features)
            valid_predict = aux_out_map['final'].data
            valid_predict_soft = torch.softmax(valid_predict, dim=-1)
            indices = torch.tensor([1])
            valid_predict_soft1 = torch.index_select(valid_predict_soft, -1, indices)

            total_loss_test = 0.0
            for name, output in aux_out_map.items():
                if name == 'final':
                    total_loss_test += nn.CrossEntropyLoss()(output, cuda_labels).item()
                else:
                    total_loss_test += a * nn.CrossEntropyLoss()(output, cuda_labels).item()

            loss_v_all += total_loss_test

            test = np.array(cuda_labels, dtype=np.float32)
            pre_p = np.array(valid_predict_soft1)
            pre = np.where(pre_p > 0.5, 1, 0)
            acc_v_all += accuracy_score(test, pre)

        loss_v_list.append(loss_v_all / len(valid_loader))
        acc_v_list.append(acc_v_all / len(valid_loader))
        # print('Epoch', e, ': valid acc', acc_v_list[-1])

        early_stopping(acc_v_all / len(valid_loader), model)
        if early_stopping.early_stop:
            # print('Early stopping')
            break

    x = range(1, len(acc_v_list) + 1)
    plt.title('valid acc')
    plt.plot(x, acc_v_list)
    plt.xlim(0, len(acc_v_list) + 1)
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.grid()
    plt.savefig('Result/train_supervision/acc_' + str(feature) + '_' + str(label_name) + '_' + str(loop))
    plt.close()

    x = range(1, len(loss_v_list) + 1)
    plt.title('valid loss')
    plt.plot(x, loss_v_list)
    plt.xlim(0, len(loss_v_list) + 1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.savefig('Result/train_supervision/loss_' + str(feature) + '_' + str(label_name) + '_' + str(loop))
    plt.close()

    return early_stopping.best_score

