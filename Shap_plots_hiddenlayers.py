import networkx as nx
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind
import seaborn as sns
import textwrap

rcParams['figure.figsize'] = [15, 10]
rcParams["figure.facecolor"] = "white"
rcParams["figure.edgecolor"] = "white"
rcParams["figure.dpi"] = 300
rcParams['lines.linewidth'] = 0.5
rcParams["font.family"] = "Arial"
rcParams['font.size'] = 6*4/3
plt.close()

knowledge = 'evidence_moderate'
feature = 'Descriptors_FCFP6_2048'
label_lists = ['Nephrotoxicity', 'Cardiotoxicity', 'Hepatotoxicity', 'Respiratory Toxicity']
event_names = pd.read_csv('Network/raw data/aop_ke_mie_ao.tsv', sep='\t', header=None)
event_names.columns = ['aop id', 'ke id', 'event class', 'event annotation']

for label in label_lists:
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', label)

    # obtain feature names
    samples = pd.read_csv('Result/Shap_subsamples/sub_' + str(label) + '_' + str(feature) + '.csv')
    labels = samples.iloc[:, 2]
    features = samples.iloc[:, 3:]
    feature_names = features.columns

    # obtain term names
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

    # load shap values : features in input layer, terms in first hidden layer, terms in all hidden layer
    shap_mean_terms = np.load('Result/Shap_values/Alllayers_seeds_AOP_' + str(label) + '.npy')
    shap_mean_terms = shap_mean_terms.swapaxes(0, 1)

    # compute shap values abs mean in all samples
    terms_shaps = pd.DataFrame({'Feature': nodes, 'SHAP Value': np.abs(shap_mean_terms[:, :, 1]).mean(0)})
    terms_shaps = terms_shaps.sort_values(by='SHAP Value', ascending=False)
    terms_shaps.to_csv('Result/Shap_values_df/' + str(label) + '_sort_alllayers.csv', index=False)

    # compute shap values abs mean in positive and negative samples
    pos_index = [j for i in np.where(samples[str(label)] == 1) for j in i]
    neg_index = [j for i in np.where(samples[str(label)] == 0) for j in i]

    terms_shaps_pos_abs = pd.DataFrame({'Feature': nodes, 'SHAP Value': np.abs(shap_mean_terms[pos_index, :, 1]).mean(0)})
    terms_shaps_negs_abs = pd.DataFrame({'Feature': nodes, 'SHAP Value': np.abs(shap_mean_terms[neg_index, :, 1]).mean(0)})
    terms_shaps_pos_abs = terms_shaps_pos_abs.sort_values(by='SHAP Value', ascending=False)
    terms_shaps_negs_abs = terms_shaps_negs_abs.sort_values(by='SHAP Value', ascending=False)
    terms_shaps_pos_abs.to_csv('Result/Shap_values_df/' + str(label) + '_sort_alllayers_pos.csv', index=False)
    terms_shaps_negs_abs.to_csv('Result/Shap_values_df/' + str(label) + '_sort_alllayers_neg.csv', index=False)

    terms_shaps_pos = pd.DataFrame({'Feature': nodes, 'SHAP Value': shap_mean_terms[pos_index, :, 1].mean(0)})
    terms_shaps_neg = pd.DataFrame({'Feature': nodes, 'SHAP Value': shap_mean_terms[neg_index, :, 1].mean(0)})
    terms_shaps_pos = terms_shaps_pos.sort_values(by='SHAP Value', ascending=False)
    terms_shaps_neg = terms_shaps_neg.sort_values(by='SHAP Value', ascending=False)
    terms_shaps_pos.to_csv('Result/Shap_values_df/' + str(label) + '_sort_alllayers_pos.csv', index=False)
    terms_shaps_neg.to_csv('Result/Shap_values_df/' + str(label) + '_sort_alllayers_neg.csv', index=False)

    print('ok')


    # draw bubble plot in all terms
    num = 5
    data = terms_shaps.iloc[0:num, :]
    data = data.sort_values(by='SHAP Value', ascending=True)
    # deep_color = {'Hepatotoxicity': (229 / 255, 139 / 255, 123 / 255),
    #               'Respiratory Toxicity': (250 / 255, 222 / 255, 188 / 255),
    #               'Nephrotoxicity': (154 / 255, 201 / 255, 219 / 255),
    #               'Cardiotoxicity': (183 / 255, 209 / 255, 205 / 255)}
    deep_color = {'Hepatotoxicity': (172 / 255, 72 / 255, 40 / 255, 0.8),
                  'Respiratory Toxicity': (172 / 255, 72 / 255, 40 / 255, 0.8),
                  'Nephrotoxicity': (172 / 255, 72 / 255, 40 / 255, 0.8),
                  'Cardiotoxicity': (172 / 255, 72 / 255, 40 / 255, 0.8)}
    colors = []
    num_shades = num * 2
    for i in range(num_shades):
        factor = i / (num_shades - 1)
        lightened_color = tuple(1 - factor * (1 - c) for c in deep_color[label])
        colors.append(lightened_color)

    plt.figure(figsize=(2.8, 2.6))
    # plt.scatter(x=data['SHAP Value'], y=data['Feature'], s=(data['SHAP Value']+0.001)*20000, color=colors[num:])
    y_values = []
    for e in data['Feature']:
        e_n = pd.DataFrame(event_names.loc[event_names['ke id'] == e, 'event annotation'])
        y_values.append(e_n.iloc[0,0])
    y_values = [textwrap.fill(y,20) for y in y_values]
    plt.scatter(x=data['SHAP Value'], y=y_values, s=(data['SHAP Value'] + 0.001) * 3000, color=colors[num:])
    plt.xlabel('Importance score', fontsize=8 * 4 / 3)
    # plt.ylabel('Event name', fontsize=8 * 4 / 3)
    plt.yticks(fontsize=7 * 4 / 3)
    plt.xticks(fontsize=6 * 4 / 3)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.title('Top 5 Key Events'+ '\n' + '('+ str(label) +')', fontsize=8 * 4 / 3)
    plt.tight_layout()
    plt.savefig('Result/Shap_plots/Bubble_plot_allterms_' + str(label)+ '.svg', dpi=300)
    plt.close()

    # # draw shap allterms summary plot in postive and negative samples
    # deep_color = {'Hepatotoxicity': (154 / 255, 201 / 255, 219 / 255),
    #               'Respiratory Toxicity': (163 / 255, 209 / 255, 205 / 255),
    #               'Nephrotoxicity': (229 / 255, 139 / 255, 123 / 255),
    #               'Cardiotoxicity': (250 / 255, 222 / 255, 188 / 255)}
    # colors = []
    # num_shades = len(nodes) * 2
    # for i in range(num_shades):
    #     factor = i / (num_shades - 1)
    #     lightened_color = tuple(1 - factor * (1 - c) for c in deep_color[label])
    #     colors.append(lightened_color)
    # width = 0.3
    # label_2 = ['Toxic', 'Non Toxic']
    # x = np.arange(len(label_2))
    # plt.figure(figsize=(15, 15))
    # i = 0
    # for l in label_2:
    #     for i in range(10):
    #         plt.barh(x + width * i, terms_shaps_pos.iloc[i, 1], width, color=colors[i])
    #         plt.barh(x + width * i, terms_shaps_neg.iloc[i, 1], width, color=colors[i])
    #
    # plt.xlabel(' ')
    # plt.yticks(fontsize=10)
    # plt.gcf().subplots_adjust(top=0.91, bottom=0.09)
    # plt.title('Importance of top terms in all layers to ' + str(label), fontsize=14)
    # plt.savefig('Result/Shap_plots/Summary_plot_allterms_' + str(label) + '.svg', dpi=300)
    # plt.close()
































