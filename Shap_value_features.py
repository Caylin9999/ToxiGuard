import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import networkx as nx
from imblearn.over_sampling import SMOTE
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from AOP_NN import AOP_NN_feature, AOP_NN_load

knowledge = 'evidence_moderate'
feature = 'Descriptors_FCFP6_2048'
label_lists = ['Hepatotoxicity', 'Respiratory Toxicity', 'Nephrotoxicity', 'Cardiotoxicity']
for label in label_lists:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', label)
    samples_traval = pd.read_csv('Data/train_' + str(label) + '.csv')

    if 'Descriptors' in str(feature):
        scaler = StandardScaler()
        scaler.fit(samples_traval.iloc[:, 3:212])
        data_traval = scaler.transform(samples_traval.iloc[:, 3:212])
        samples_traval[samples_traval.columns[3:212]] = pd.DataFrame(data_traval, columns=list(samples_traval.columns)[3:212])
        samples_traval = samples_traval.fillna(0)

    samples_traval = samples_traval.sample(n=500, random_state=42)
    # samples_traval.to_csv('Result/Shap_subsamples/sub_' + str(label) + '_' + str(feature) + '.csv', index=False)

    trainval_x = np.array(samples_traval.iloc[:, 3:])
    trainval_y = samples_traval.loc[:, str(label)].to_numpy()
    trainval_x = torch.tensor(trainval_x).float()

    dG = nx.read_gml('Network/dG/' + str(knowledge) + '/dG_' + str(label) + '.gml')
    layers = []
    nodes = []
    while True:
        leaves = [n for n in dG.nodes() if dG.in_degree(n) == 0]
        if len(leaves) == 0:
            break
        layers.append(leaves)
        for n in leaves:
            nodes.append(n)
        dG.remove_nodes_from(leaves)
    nodes = nodes[:-1]

    for seed in range(5):

        loop = 0
        shap_loops = []
        shap_loops_features = []

        for i in range(5):
            train_sub = trainval_x
            dG = nx.read_gml('Network/dG/' + str(knowledge) + '/dG_' + str(label) + '.gml')
            model_orig = torch.load('Model/' + str(knowledge) + '/' + str(seed) + '_' + str(feature) + '_' + str(label) + '_' + str(loop) + '.pth')
            aux_out_map, term_NN_out_map = model_orig(train_sub)

            model_feature = AOP_NN_feature(dG, 2257, 6)
            state_dict_explain = {k: model_orig.state_dict()[k] for k in model_feature.state_dict().keys()}
            model_feature.load_state_dict(state_dict_explain)
            explainer_features = shap.DeepExplainer(model_feature, train_sub)
            shap_values_feature = explainer_features.shap_values(train_sub)
            shap_loops_features.append(shap_values_feature)

            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', loop)
            loop += 1
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~seed', seed)

        shap_loops_mean_features = np.mean(np.array(shap_loops_features), axis=0)  # n_sample, 2257, 2
        print(shap_loops_mean_features.shape)
        np.save('Result/Shap_values/Features_samples_AOP_' + str(label) + '_sub_' + str(seed) + '.npy',
                shap_loops_mean_features)

print('ok')

