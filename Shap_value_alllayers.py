import numpy as np
import pandas as pd
import shap
import torch
import networkx as nx
from sklearn.preprocessing import StandardScaler
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
    train_x = torch.tensor(trainval_x).float()

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

        for i in range(5):
            train_sub = train_x
            dG = nx.read_gml('Network/dG/' + str(knowledge) + '/dG_' + str(label) + '.gml')
            # Obtain the output and representation of each node
            model_orig = torch.load('Model/' + str(knowledge) + '/' + str(seed) + '_' + str(feature) + '_' + str(label) + '_' + str(loop) + '.pth')
            aux_out_map, term_NN_out_map = model_orig(train_sub)

            # construct the intermediate input of first layer
            l = len(layers) - 1
            layers_delete = [e for layer in layers[0: l] for e in layer]
            model_explain = AOP_NN_load(layers_delete, dG, 2257, 6)
            state_dict_explain = {k: model_orig.state_dict()[k] for k in model_explain.state_dict().keys()}
            model_explain.load_state_dict(state_dict_explain)

            inputdata = {}
            inputdata_list = []
            terms_dim = []
            for e in layers_delete:
                terms_dim.append(model_orig.term_dim_map[e])
                inputdata[e] = term_NN_out_map[e]
                inputdata_list.append(term_NN_out_map[e])
            inputdata_tensor = torch.cat(inputdata_list, 1)

            explainer = shap.DeepExplainer(model_explain, inputdata_tensor)
            shap_values = explainer.shap_values(inputdata_tensor)

            shap_terms = []
            m, n = 0, 0
            for t, e in enumerate(layers_delete):
                n += terms_dim[t]
                shap_term = np.mean(shap_values[:, m:n, :], axis=1)
                shap_terms.append(shap_term)
                m += terms_dim[t]

            shap_loops.append(shap_terms)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', loop)
            loop += 1
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~seed', seed)

        shap_loops_terms = np.array(shap_loops)
        shap_loops_mean = np.mean(shap_loops_terms, axis=0)  # terms, n_sample, 2
        print(shap_loops_mean.shape)
        np.save('Result/Shap_values/Alllayers_samples_AOP_' + str(label) + '_sub_' + str(seed) + '.npy',
                shap_loops_mean)

print('ok')




