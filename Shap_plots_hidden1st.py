import textwrap

import networkx as nx
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_ind
import seaborn as sns


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

label_lists = ['Nephrotoxicity', 'Hepatotoxicity', 'Respiratory Toxicity', 'Cardiotoxicity']
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
    shap_mean_first = np.load('Result/Shap_values/Firstlayer_seeds_AOP_' + str(label) + '.npy')
    shap_mean_first = shap_mean_first.swapaxes(0, 1)

    # compute shap values abs mean in all samples
    first_shaps = pd.DataFrame({'Feature': layers[0], 'SHAP Value': np.abs(shap_mean_first[:, :, 1]).mean(0)})
    first_shaps_rank = first_shaps.sort_values(by='SHAP Value', ascending=False)
    first_shaps_rank.to_csv('Result/Shap_values_df/' + str(label) + '_sort_firstlayer.csv', index=False)

    # # compute shap values abs mean in positive and negative samples
    # pos_index = [j for i in np.where(samples[str(label)] == 1) for j in i]
    # neg_index = [j for i in np.where(samples[str(label)] == 0) for j in i]
    #
    # first_shaps_pos = pd.DataFrame({'Feature': layers[0], 'SHAP Value_pos': np.abs(shap_mean_first[pos_index, :, 1]).mean(0)})
    # first_shaps_neg = pd.DataFrame({'Feature': layers[0], 'SHAP Value_neg': np.abs(shap_mean_first[neg_index, :, 1]).mean(0)})
    # first_shaps_pos = first_shaps_pos.sort_values(by='SHAP Value_pos', ascending=False)
    # first_shaps_neg = first_shaps_neg.sort_values(by='SHAP Value_neg', ascending=False)
    # first_shaps_pos.to_csv('Result/Shap_values_df/' + str(label) + '_sort_firstlayer_pos.csv', index=False)
    # first_shaps_neg.to_csv('Result/Shap_values_df/' + str(label) + '_sort_firstlayer_neg.csv', index=False)
    #
    # if label != 'Cardiotoxicity':
    #     first_shaps_pos_neg = pd.merge(first_shaps_pos, first_shaps_neg, on=['Feature'], how='left')[:10]
    #     plt.figure(figsize=(15, 10))
    # else:
    #     first_shaps_pos_neg = pd.merge(first_shaps_pos, first_shaps_neg, on=['Feature'], how='left')
    #     plt.figure(figsize=(15, 10))
    #
    # print('ok')
    #
    # # bar plot in positive and negative samples
    # deep_color = {'Hepatotoxicity': (229 / 255, 139 / 255, 123 / 255),
    #               'Respiratory Toxicity': (250 / 255, 222 / 255, 188 / 255),
    #               'Nephrotoxicity': (154 / 255, 201 / 255, 219 / 255),
    #               'Cardiotoxicity': (183 / 255, 209 / 255, 205 / 255)}
    # colors = []
    # num_shades = len(first_shaps_pos_neg['SHAP Value_pos'].tolist()) * 2
    # for i in range(num_shades):
    #     factor = i / (num_shades - 1)
    #     lightened_color = tuple(1 - factor * (1 - c) for c in deep_color[label])
    #     colors.append(lightened_color)
    # bar_positions = np.arange(len(first_shaps_pos_neg['SHAP Value_pos'].tolist()))
    #
    # plt.scatter(bar_positions, np.abs(first_shaps_pos_neg['SHAP Value_pos']),
    #             s=100, label='Non Toxic', color=colors[len(first_shaps_pos_neg['SHAP Value_pos'].tolist()):])
    # plt.plot(bar_positions, first_shaps_pos_neg['SHAP Value_pos'], c=deep_color[label], linewidth=4.0)
    # plt.scatter(bar_positions, -first_shaps_pos_neg['SHAP Value_neg'],
    #             s=100, label='Toxic', color='lightgrey', alpha=0.8)
    # plt.plot(bar_positions, -first_shaps_pos_neg['SHAP Value_neg'], c='lightgrey', linewidth=4.0)
    # for a, b, c in zip(bar_positions+0.001, first_shaps_pos_neg['SHAP Value_pos'], first_shaps_pos_neg['SHAP Value_neg']):
    #     plt.text(a, b+0.001, '%.4f' % b, ha='center', va='bottom', fontsize=13)
    #     plt.text(a, -c-0.001, '%.4f' % c, ha='center', va='bottom', fontsize=13)
    # plt.xticks(bar_positions, first_shaps_pos_neg['Feature'], rotation=0, fontsize=20)
    # plt.yticks(fontsize=10)
    # plt.ylabel('SHAP values', fontsize=20)
    # plt.gcf().subplots_adjust(top=0.91, bottom=0.09)
    # plt.gca().spines['top'].set_linewidth(0)
    # plt.gca().spines['right'].set_linewidth(0)
    # plt.gca().spines['bottom'].set_linewidth(1.5)
    # plt.gca().spines['left'].set_linewidth(1.5)
    # plt.gca().spines['bottom'].set_position(('data', 0))
    #
    # plt.legend(handles=[plt.Rectangle((0, 0), 0, 1, color=color, label=label) for color, label in
    #                     zip(['lightgrey', deep_color[label]], ['Non Toxic', 'Toxic'])],
    #            fontsize=15, loc='lower right')
    # plt.title('Importance of MIEs to ' + str(label), fontsize=25)
    # plt.savefig('Result/Shap_plots/Barh_plot_first_' + str(label) + '.svg', dpi=300)
    # plt.close()

    # draw bubble plot in all terms
    if len(first_shaps_rank)<5:
        num=len(first_shaps_rank)
        plt.figure(figsize=(2.8, 2))
        s_d = -10
    else:
        num = 5
        plt.figure(figsize=(2.8, 3.4))
        s_d = 20
    data = first_shaps_rank.iloc[0:num, :]
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

    # plt.scatter(x=data['SHAP Value'], y=data['Feature'], s=(data['SHAP Value']+0.001)*20000, color=colors[num:])
    y_values = []
    for e in data['Feature']:
        e_n = pd.DataFrame(event_names.loc[event_names['ke id'] == e, 'event annotation'])
        y_values.append(e_n.iloc[0, 0])
    y_values = [textwrap.fill(y, 20) for y in y_values]
    x_values = data['SHAP Value'].tolist()
    c = colors[num:]
    plt.scatter(x=data['SHAP Value'], y=y_values, s=data['SHAP Value']*1000+s_d, color=colors[num:])
    plt.xlabel('Importance score', fontsize=8 * 4 / 3)
    # plt.ylabel('Event name', fontsize=8 * 4 / 3)
    plt.yticks(fontsize=7 * 4 / 3)
    plt.xticks(fontsize=6 * 4 / 3)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.title('Top '+str(num)+' MIEs' + '\n' + '(' + str(label) + ')''\n' +' ', fontsize=8 * 4 / 3)
    plt.tight_layout()
    plt.savefig('Result/Shap_plots/Barh_plot_first_' + str(label) + '.svg', dpi=300)
    plt.close()


    # heatmap in top terms
    # cmap_list = {'Hepatotoxicity': 'coolwarm',
    #              'Respiratory Toxicity': 'pink_r',
    #              'Nephrotoxicity': 'BuGn',
    #              'Cardiotoxicity': 'Greens'}
    cmap_list = {'Hepatotoxicity': 'GnBu',
                 'Respiratory Toxicity': 'GnBu',
                 'Nephrotoxicity': 'GnBu',
                 'Cardiotoxicity': 'GnBu'}
    plt.figure(figsize=(2, 1.5))
    shap_mean_first = pd.DataFrame(shap_mean_first[:, :, 1], columns=layers[0])
    shap_mean_first.insert(len(shap_mean_first.columns.tolist()), 'label', labels.tolist())
    mean_shap = first_shaps['SHAP Value'].tolist()
    mean_shap.append(5)
    shap_mean_first.loc[len(shap_mean_first.index)] = mean_shap
    shap_mean_first = shap_mean_first.sort_values(by='label', ascending=True)
    shap_mean_first = shap_mean_first.drop(columns=['label'])
    shap_mean_first = shap_mean_first.T
    shap_mean_first = shap_mean_first.sort_values(by=shap_mean_first.columns[-1], ascending=False)
    shap_mean_first = shap_mean_first.drop(columns=shap_mean_first.columns[-1])
    data = shap_mean_first.iloc[:5, :]
    data.index = y_values[:5]

    ax = sns.heatmap(data, cmap=cmap_list[label])
    plt.xlabel('Non toxic -> Toxic', fontsize=8 * 4 / 3)
    plt.xticks([], [])
    plt.yticks([], [])
    # plt.title(str(y_values[0]) +'\n'+ '('+str(label)+')', fontsize=8 * 4 / 3)
    plt.title(str(label), fontsize=8 * 4 / 3)
    plt.tight_layout()
    plt.savefig('Result/Shap_plots/Heatmap_plot_first_' + str(label) + '.svg', dpi=300)
    plt.close()


























