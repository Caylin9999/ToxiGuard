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
#
# label_lists = ['Nephrotoxicity', 'Cardiotoxicity', 'Respiratory Toxicity', 'Hepatotoxicity']
# # features = {'Nephrotoxicity': ['SlogP_VSA11'], 'Cardiotoxicity': ['SlogP_VSA10', 'NumRotatableBonds'],
# #             'Respiratory Toxicity': ['MolLogP', 'FractionCSP3'], 'Hepatotoxicity': ['SlogP_VSA11']}
# for l in label_lists:
#     f_df = pd.read_csv(str(l)+'_sort_features.csv')
#     values = f_df.loc[f_df['Feature'] == features[l], 'SHAP Value']



# pie plot
types = ['Charge distribution', 'Chemical group',  'Drug-like properties', 'Stereochemistry']
values = [[4, 4,2], [4,4,1,1], [3,1,5,1], [7, 2, 1]]
# colors =[(123/255, 209/255, 203/255, 0.5), (182 / 255, 226 / 255, 220 / 255, 0.5),
#          (132 / 255, 204 / 255, 224 / 255, 0.5), (172 / 255, 226 / 255, 190 / 255, 0.5),]
colors =[(58/255, 181/255, 179/255, 0.5), (156 / 255, 180 / 255, 153 / 255, 0.5), (99 / 255, 173 / 255, 139 / 255, 0.5),
         (104 / 255, 140 / 255, 168 / 255, 0.5),]
explodes =[0,0,0,0]
# plt.pie(values, labels=types, colors=colors, startangle=180, shadow=True, explode=explodes, autopct='%1.1f%%')
plt.figure(figsize=(3, 4))
plt.pie(values[2], radius=1.1, colors=colors, labeldistance=1.2, pctdistance=0.85, startangle=290,
        wedgeprops=dict(width=0.3, edgecolor="w"), autopct='%1.0f%%', explode=explodes)
plt.pie(values[1], radius=0.8, colors=colors, labeldistance=1.2, pctdistance=0.8, startangle=290,
        wedgeprops=dict(width=0.3, edgecolor="w"), autopct='%1.0f%%', explode=explodes)
plt.legend(types, loc=(0.3, -0.4))
patches, texts, autotexts = plt.pie(values[3],radius=1.4,colors=colors,labeldistance=1.2, pctdistance=0.9,startangle=290,wedgeprops=dict(width=0.3,edgecolor="w"), autopct='%1.0f%%', explode=explodes[:3])
# autotexts[0].set_color('b')
plt.pie(values[0],radius=0.5, colors=colors,labeldistance=1.2, pctdistance=0.7,startangle=290,wedgeprops=dict(width=0.3,edgecolor="w"), autopct='%1.0f%%', explode=explodes[:3])


plt.rcParams["font.family"] = "Arial"
plt.rcParams['font.size'] = 6 * 4 / 3
plt.tight_layout()
plt.savefig('Result/Shap_plots/Pie_plot_desfeatures.svg', dpi=300)
plt.close()


knowledge = 'evidence_moderate'
feature = 'Descriptors_FCFP6_2048'
label_lists = ['Nephrotoxicity', 'Cardiotoxicity', 'Hepatotoxicity', 'Respiratory Toxicity']
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
    shap_mean_features = np.load('Result/Shap_values/Features_seeds_AOP_' + str(label) + '.npy')

    # compute shap values abs mean in all samples
    features_shaps = pd.DataFrame({'Feature': feature_names, 'SHAP Value': np.abs(shap_mean_features[:, :, 1]).mean(0)})
    features_shaps = features_shaps.sort_values(by='SHAP Value', ascending=False)
    features_shaps.to_csv('Result/Shap_values_df/' + str(label) + '_sort_features.csv', index=False)


    # compute shap values abs mean in positive and negative samples
    pos_index = [j for i in np.where(samples[str(label)] == 1) for j in i]
    neg_index = [j for i in np.where(samples[str(label)] == 0) for j in i]

    features_shaps_pos = pd.DataFrame({'Feature': feature_names, 'SHAP Value': np.abs(shap_mean_features[pos_index, :, 1]).mean(0)})
    features_shaps_neg = pd.DataFrame({'Feature': feature_names, 'SHAP Value': np.abs(shap_mean_features[neg_index, :, 1]).mean(0)})
    features_shaps_pos = features_shaps_pos.sort_values(by='SHAP Value', ascending=False)
    features_shaps_neg = features_shaps_neg.sort_values(by='SHAP Value', ascending=False)
    features_shaps_pos.to_csv('Result/Shap_values_df/' + str(label) + '_sort_features_pos.csv', index=False)
    features_shaps_neg.to_csv('Result/Shap_values_df/' + str(label) + '_sort_features_neg.csv', index=False)

    print('ok')

    # plot bars of all features
    plt.figure(figsize=(3, 2.5))
    plt.bar(x=np.arange(len(features_shaps)), height=features_shaps.iloc[:, 1],
            color=(149 / 255, 139 / 255, 191 / 255), width=1)
    # plt.ylim(0, 0.8e-9)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.yticks(fontsize=6 * 4 / 3)
    plt.title('The feature ranking in ' + str(label), fontsize=8 * 4 / 3)
    plt.ylabel('Importance score', fontsize=8 * 4 / 3)
    plt.xlabel('Feature ranking', fontsize=8 * 4 / 3)
    plt.text(400, plt.ylim()[1]*5/7, str('top 10%'), fontsize=8 * 4 / 3)
    plt.text(400, plt.ylim()[1]*4/7, str('contribution radio: ')+'{:.1f}%'.format(features_shaps.iloc[:int(0.1*len(features_shaps)), 1].sum()/features_shaps.iloc[:, 1].sum()*100), fontsize=8 * 4 / 3)
    plt.tight_layout()
    # plt.gcf().subplots_adjust(top=0.91, bottom=0.09)
    plt.savefig('Result/Shap_plots/Summary_plot_allfeatures_' + str(label) + '.svg', dpi=300)
    plt.close()

    # plot bars of sub features
    plt.figure(figsize=(2.5, 2))
    plt.bar(x=np.arange(len(features_shaps.iloc[209:, :])), height=features_shaps.iloc[209:, 1],
            color=(123/255, 209/255, 203/255, 0.7), width=1)
    # plt.ylim(0, 0.8e-9)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.yticks(fontsize=6 * 4 / 3)
    plt.ylabel('Importance score', fontsize=8 * 4 / 3)
    plt.tight_layout()
    # plt.gcf().subplots_adjust(top=0.91, bottom=0.09)
    plt.savefig('Result/Shap_plots/Summary_plot_subfeatures_' + str(label) + '.svg', dpi=300)
    plt.close()

    # plot bars of des features
    plt.figure(figsize=(2.5, 2))
    plt.bar(x=np.arange(len(features_shaps.iloc[:209, :])), height=features_shaps.iloc[:209, 1], color=(123/255, 209/255, 203/255, 0.7), width=1)
    # plt.ylim(0, 0.8e-9)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.yticks(fontsize=6 * 4 / 3)
    plt.ylabel('Importance score', fontsize=8 * 4 / 3)
    plt.tight_layout()
    # plt.gcf().subplots_adjust(top=0.91, bottom=0.09)
    plt.savefig('Result/Shap_plots/Summary_plot_desfeatures_' + str(label) + '.svg', dpi=300)
    plt.close()


    # plot dots of descriptors
    shap.summary_plot(shap_mean_features[:, : 209, 1], features.iloc[:, :209], cmap='GnBu', feature_names=feature_names[: 209], max_display=10, plot_type='dot', show=False, color_bar_label=' ')
    fig = plt.gcf()
    fig.set_size_inches(4.5, 3.2)
    plt.rcParams["font.family"]="Arial"
    plt.rcParams['font.size'] = 6*4/3
    plt.xticks(fontsize=6 * 4 / 3)
    plt.yticks(fontsize=8*4/3)
    plt.xlabel('Importance score', fontsize=8*4/3)
    plt.title('Top 10 descriptors in ' + str(label)+str('        '), fontsize=8 * 4 / 3)
    plt.tight_layout()
    plt.savefig('Result/Shap_plots/Summary_plot_descriptors_' + str(label) + '.svg', dpi=300)
    plt.close()


    # plot dots of substructures
    shap.summary_plot(shap_mean_features[:, 210:, 1], features.iloc[:, 210:], feature_names=feature_names[210:], cmap=cm.get_cmap('Pastel2_r', 2), max_display=10, plot_type='dot', show=False, color_bar_label=' ')
    fig = plt.gcf()
    fig.set_size_inches(4.5, 3.2)
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 6 * 4 / 3
    plt.title('Top 10 FPs in ' + str(label), fontsize=8 * 4 / 3)
    plt.yticks(fontsize=8*4/3)
    plt.xticks(fontsize=6 * 4 / 3)
    plt.xlabel('Importance score', fontsize=8*4/3)
    plt.tight_layout()
    plt.savefig('Result/Shap_plots/Summary_plot_substructures_' + str(label) + '.svg', dpi=300)
    plt.close()


    # descriptor dependence plot
    list_disp = {'Nephrotoxicity': 'SlogP_VSA11', 'Cardiotoxicity': 'NumRotatableBonds',
                 'Respiratory Toxicity': 'FractionCSP3', 'Hepatotoxicity': 'SlogP_VSA11'}
    ind = features.columns.get_loc(list_disp[label])
    plt.figure(figsize=(2, 2))
    plt.scatter(features.loc[:, list_disp[label]], shap_mean_features[:, features.columns.get_loc(list_disp[label]), 1], s=10, alpha=1, c=(196 / 255, 194 / 255, 218 / 255, 0.7))
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['font.size'] = 6 * 4 / 3
    plt.title(str(label)+'\n' +'('+ str(list_disp[label])+')', fontsize=8 * 4 / 3)
    plt.yticks(fontsize=6*4/3)
    plt.xticks(fontsize=6 * 4 / 3)
    plt.ylabel('Importance score', fontsize=8*4/3)
    plt.xlabel('Descriptor value', fontsize=8 * 4 / 3)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig('Result/Shap_plots/Dependence_plot_' + str(label) + '.svg', dpi=300)
    plt.close()

    # substructure dependence plot
    list_disp = {'Respiratory Toxicity': '1', 'Hepatotoxicity': '1556', 'Nephrotoxicity': '2017', 'Cardiotoxicity': '546'}
    plt.figure(figsize=(2, 2))

    plt.scatter(features.loc[:, list_disp[label]], shap_mean_features[:, features.columns.get_loc(list_disp[label]), 1],
                s=5, alpha=1, c=(156 / 255, 136 / 255, 182 / 255, 0.5))
    xy_df = pd.concat([features.loc[:, list_disp[label]], pd.DataFrame(shap_mean_features[:, features.columns.get_loc(list_disp[label]), 1])], axis=1)
    xy_df.columns = ['FP value', 'shap value']
    mean = xy_df.groupby('FP value')['shap value'].mean()
    # sns.swarmplot(x=xy_df['FP value'], y=xy_df['shap value'],  c=(182 / 255, 226 / 255, 220 / 255, 0.7), s=3, alpha=1, linewidth=0.5)
    plt.plot([-0.25, 0.25], [mean[0], mean[0]], color='purple', linestyle='--', linewidth=0.5)
    plt.text(-0.2, mean[0]+0.01, '%.2f' %mean[0], fontsize=8 * 4 / 3)
    plt.plot([0.75, 1.25], [mean[1], mean[1]], color='purple', linestyle='--', linewidth=0.5)
    plt.text(0.8, mean[1]+0.01, '%.2f' %mean[1], fontsize=8 * 4 / 3)
    # plt.plot(features.loc[:, list_disp[label]], shap_mean_features[:, features.columns.get_loc(list_disp[label]), 1].mean(), color='g', linestyle='--', linewidth=1)
    plt.rcParams["font.family"] = "Arial"
    plt.title(str(label) + '\n' + '(FP' + str(list_disp[label]) + ')', fontsize=8 * 4 / 3)
    plt.yticks(fontsize=6 * 4 / 3)
    plt.xticks(fontsize=6 * 4 / 3)
    plt.ylabel('Importance score', fontsize=8 * 4 / 3)
    plt.xlabel('Fingerprint value', fontsize=8 * 4 / 3)
    plt.gca().spines['top'].set_linewidth(0)
    plt.gca().spines['right'].set_linewidth(0)
    plt.gca().spines['bottom'].set_linewidth(0.5)
    plt.gca().spines['left'].set_linewidth(0.5)
    plt.tight_layout()
    plt.rcParams['font.size'] = 6 * 4 / 3
    plt.rcParams["font.family"] = "Arial"
    plt.savefig('Result/Shap_plots/Dependence_sub_plot_' + str(label) + '.svg', dpi=300)
    plt.close()

























