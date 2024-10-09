import random
import pandas as pd
import numpy as np
import scipy
import torch
import argparse
import os
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from Train import train
from predict import Predict
import networkx as nx
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

label_lists = ['Hepatotoxicity', 'Respiratory Toxicity', 'Nephrotoxicity', 'Cardiotoxicity']
metrixs = ['AUROC', 'Accuracy', 'AUPR', 'BACC', 'F1 score', 'Precision', 'Recall', 'MCC']


def Main(seed, target_label):
    a = 0.2

    batch_size = {'Respiratory Toxicity': 224, 'Hepatotoxicity': 160, 'Nephrotoxicity': 44, 'Cardiotoxicity': 160}
    learning_rate = {'Respiratory Toxicity': 3e-05, 'Hepatotoxicity': 4e-05, 'Nephrotoxicity': 4.5e-05, 'Cardiotoxicity': 9e-05}

    parser = argparse.ArgumentParser(description='PyTorch implementation of AOP_Tox')
    parser.add_argument('--label', type=str, default='Respiratory Toxicity', help='which dataset to use')
    parser.add_argument('--feature', type=str, default='Descriptors_FCFP6_2048', help='which feature to use')
    parser.add_argument('--knowledge', type=str, default='evidence_moderate', help='which knowledge to use')
    parser.add_argument('--device', type=int, default=0, help='which gpu to use if any (default: 0)')
    parser.add_argument('--runseed', type=int, default=0, help='Seed for minibatch selection, random initialization.')
    parser.add_argument('--num_loop', type=int, default=5, help='number of loop')
    parser.add_argument('--epoch', type=int, default=300, help='number of epoch to train (default: 200)')
    parser.add_argument('--input_dim', type=int, default=167, help='input dimensions (default: 167)')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--num_hiddens_ratio', type=int, default=6, help='')
    parser.add_argument('--learning_rate', type=float, default=1e-05, help='')

    args = parser.parse_args()

    random.seed(args.runseed)
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    torch.cuda.manual_seed_all(args.runseed)

    args.label = target_label
    args.batch_size = batch_size[args.label]
    args.learning_rate = learning_rate[args.label]

    # input data
    samples_traval = pd.read_csv('Data/train_' + str(args.label) + '.csv')
    samples_test = pd.read_csv('Data/test_' + str(args.label) + '.csv')

    print('train data info: ', samples_traval.loc[:, args.label].value_counts())
    print('test data info:', samples_test.loc[:, args.label].value_counts())

    if 'Descriptors' in str(args.feature):
        scaler = StandardScaler()
        scaler.fit(samples_traval.iloc[:, 3:212])
        data_traval = scaler.transform(samples_traval.iloc[:, 3:212])
        data_test = scaler.transform(samples_test.iloc[:, 3:212])
        samples_traval[samples_traval.columns[3:212]] = pd.DataFrame(data_traval, columns=list(samples_traval.columns)[3:212])
        samples_test[samples_test.columns[3:212]] = pd.DataFrame(data_test, columns=list(samples_test.columns)[3:212])
        samples_traval = samples_traval.fillna(0)
        samples_test = samples_test.fillna(0)

    data = np.array(samples_traval.iloc[:, 3:])
    label = samples_traval.loc[:, args.label].to_numpy()
    args.input_dim = len(samples_traval.iloc[:, 3:].columns)

    loop = 0
    val_acc_list = []
    skf = StratifiedKFold(n_splits=args.num_loop, shuffle=True, random_state=seed)
    for train_index, valid_index in skf.split(data, label):
        train_x, train_y = data[train_index], label[train_index]
        train_x, train_y = SMOTE(random_state=args.runseed).fit_resample(train_x, train_y)
        valid_x, valid_y = data[valid_index], label[valid_index]
        train_x = torch.tensor(train_x).float()
        train_y = torch.LongTensor(train_y)
        valid_x = torch.tensor(valid_x).float()
        valid_y = torch.LongTensor(valid_y)

        traindata = TensorDataset(train_x, train_y)
        validdata = TensorDataset(valid_x, valid_y)

        dG = nx.read_gml('Network/dG/' + str(args.knowledge) + '/dG_' + str(args.label) + '.gml')

        val_acc = train(seed, a, args.batch_size, loop, args.epoch,  args.label,  args.feature,  args.learning_rate, dG,  args.input_dim,  args.num_hiddens_ratio, traindata, validdata, args.knowledge)
        val_acc_list.append(val_acc)
        loop += 1
    print('---------------------------------------------------------------------seed:', seed, 'val acc', np.mean(np.array(val_acc_list)))

    results = Predict(seed, args.num_loop, args.label, args.feature, samples_test, args.knowledge)
    return results, args.knowledge, args.feature


if __name__ == '__main__':

    times = 5

    results_times_mean = []
    results_times_std = []
    results_times = []
    for target_label in label_lists:

        label_times = []
        for seed in range(times):
            results, KF, feature = Main(seed, target_label)
            label_times.append(results.loc['VNN', :].tolist())

        label_times = pd.DataFrame(label_times, columns=metrixs)
        label_times.to_csv('Result/Performance comparison/Times_ToxiGuard AOPNN_'+str(target_label)+'.csv')

        label_mean = [np.mean(label_times.loc[:, i]) for i in label_times.columns]
        label_std = [np.std(label_times.loc[:, i]) for i in label_times.columns]

        results_times_mean.append(label_mean)
        results_times_std.append(label_std)

    results_times_mean = pd.DataFrame(results_times_mean, columns=metrixs, index=label_lists).round(3)
    results_times_std = pd.DataFrame(results_times_std, columns=metrixs, index=label_lists).round(3)
    results_times_mean.to_csv('Result/Times_ToxiGuard AOPNN_metrix_mean.csv')
    results_times_std.to_csv('Result/Times_ToxiGuard AOPNN_metrix_std.csv')

    print('----------------------------------------------------------------', times, 'times results: ', results_times_mean)
