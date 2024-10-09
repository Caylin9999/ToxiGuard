import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

label_lists = ['Respiratory Toxicity', 'Hepatotoxicity', 'Nephrotoxicity', 'Cardiotoxicity']

## Cardiotoxicity dataset
# 0.input raw data
samples_card = pd.read_excel('Data/raw data/raw/hERG_all.xlsx').iloc[:, [1, 5]]  # 7889
samples_card = samples_card.dropna()  # 5161

# 1.tranform smiles to canonical smiles
mol_card = [Chem.MolFromSmiles(x) for x in samples_card.iloc[:, 0]]
smiles_card = []
for mol in mol_card:
    if mol:
        smiles_card.append(Chem.MolToSmiles(mol))
    else:
        smiles_card.append(np.nan)
samples_card.iloc[:, 0] = smiles_card
samples_card = samples_card.dropna()  # 5134
# Results: sample 5134


# 3.delete unconsistant sample to label
cols = samples_card.columns
duplicate_smiles = samples_card[samples_card.duplicated(subset=[cols[0]], keep=False)]  # 186
duplicate_smiles_list = duplicate_smiles[cols[0]].unique()  # 81
indexes_to_drop = set()  # 37
smiles_to_inconsist = []  # 13
for smiles in duplicate_smiles[cols[0]].unique():
    rows = duplicate_smiles[duplicate_smiles[cols[0]] == smiles]
    if len(rows[cols[1]].unique()) > 1:
        indexes_to_drop.update(rows.index)
        smiles_to_inconsist.append(smiles)
samples_card = samples_card.drop(index=indexes_to_drop)  # 5097
# Results: sample 5097


# 4. delete duplicate samples in smiles
samples_card = samples_card.drop_duplicates() # 5016
smiles_unique = samples_card.iloc[:, 0].unique() # 5016


# save samples
samples_traval, samples_test = train_test_split(samples_card, test_size=0.2, stratify=samples_card.iloc[:, 1], shuffle=True, random_state=33)
samples_test.columns = ['Canonical SMILES', 'Cardiotoxicity']
samples_traval.columns = ['Canonical SMILES', 'Cardiotoxicity']
samples_card.columns = ['Canonical SMILES', 'Cardiotoxicity']
samples_card.to_csv('Data/raw data/Cardiotoxicity_all.csv', index=False)
samples_test.to_csv('Data/raw data/Cardiotoxicity_test.csv', index=False)
samples_traval.to_csv('Data/raw data/Cardiotoxicity_train.csv', index=False)
# Results: sample 5016, train 4012, test 1004



## Respiratory Toxicity dataset
# 0.input raw data
samples_traval_resp = pd.read_csv('Data/raw data/raw/Respiratory toxicity_0_train.csv')  # 2336
samples_test_resp = pd.read_csv('Data/raw data/raw/Respiratory toxicity_0_test.csv')  # 193

# 1.tranform smiles to canonical smiles
mol_traval_resp = [Chem.MolFromSmiles(x) for x in samples_traval_resp.iloc[:, 0]]
smiles_traval_resp = [Chem.MolToSmiles(x) for x in mol_traval_resp]
samples_traval_resp.iloc[:, 0] = smiles_traval_resp

mol_test_resp = [Chem.MolFromSmiles(x) for x in samples_test_resp.iloc[:, 0]]
smiles_test_resp = [Chem.MolToSmiles(x) for x in mol_test_resp]
samples_test_resp.iloc[:, 0] = smiles_test_resp


# 2.delete traval samples in test dataset
smiles_inter = list(set(samples_traval_resp.iloc[:, 0]) & set(samples_test_resp.iloc[:, 0]))
samples_traval_resp = samples_traval_resp[~samples_traval_resp.iloc[:, 0].isin(smiles_inter)]
print(len(smiles_inter))  # 0
# Results: traval sample 2336, test sample 193


# 3.delete unconsistant sample to label
cols = samples_traval_resp.columns
duplicate_smiles = samples_traval_resp[samples_traval_resp.duplicated(subset=[cols[0]], keep=False)]  # 0
duplicate_smiles_list = duplicate_smiles[cols[0]].unique()  # 46
indexes_to_drop = set()  # 5
smiles_to_inconsist = []  # 2
for smiles in duplicate_smiles[cols[0]].unique():
    rows = duplicate_smiles[duplicate_smiles[cols[0]] == smiles]
    if len(rows[cols[1]].unique()) > 1:
        indexes_to_drop.update(rows.index)
        smiles_to_inconsist.append(smiles)
samples_traval_resp = samples_traval_resp.drop(index=indexes_to_drop)  # 2336
# Results: traval sample 2336, test sample 193


# 4. delete duplicate samples in smiles
samples_traval_resp = samples_traval_resp.drop_duplicates()
smiles_unique_traval = samples_traval_resp.iloc[:, 0].unique()  # 2336

samples_test_resp = samples_test_resp.drop_duplicates()
smiles_unique_test = samples_test_resp.iloc[:, 0].unique() # 193
# Results: traval sample 2336, test sample 193


# save samples
samples_traval_resp.columns = ['Canonical SMILES', 'Respiratory Toxicity']
samples_test_resp.columns = ['Canonical SMILES', 'Respiratory Toxicity']
samples_traval_resp.to_csv('Data/raw data/Respiratory Toxicity_train.csv', index=False)
samples_test_resp.to_csv('Data/raw data/Respiratory Toxicity_test.csv', index=False)



## Nephrotoxicity dataset
# 0.input raw data
samples_neph = pd.read_csv('Data/raw data/raw/Nephrotoxicity_raw.csv').iloc[:, [3, 1]]  # 565
label_list = []
for i in range(len(samples_neph)):
    if samples_neph.iloc[i, 1] == 'nephrotoxic':
        label_list.append(1)
    else:
        label_list.append(0)
samples_neph.iloc[:, 1] = label_list

# 1.tranform smiles to canonical smiles
mol_neph = [Chem.MolFromSmiles(x) for x in samples_neph.iloc[:, 0]]
smiles_neph = [Chem.MolToSmiles(x) for x in mol_neph]
samples_neph.iloc[:, 0] = smiles_neph

# 2.no duplicate samples in smiles
smiles_neph_unique = samples_neph.iloc[:, 0].unique()  # 565

# 3. save samples
samples_traval, samples_test = train_test_split(samples_neph, test_size=0.2, stratify=samples_neph.iloc[:, 1], shuffle=True, random_state=22)
samples_neph.columns = ['Canonical SMILES', 'Nephrotoxicity']
samples_test.columns = ['Canonical SMILES', 'Nephrotoxicity']
samples_traval.columns = ['Canonical SMILES', 'Nephrotoxicity']
samples_neph.to_csv('Data/raw data/Nephrotoxicity_all.csv', index=False) # 565
samples_test.to_csv('Data/raw data/Nephrotoxicity_test.csv', index=False) # 113
samples_traval.to_csv('Data/raw data/Nephrotoxicity_train.csv', index=False) # 452



## Hepatotoxicity dataset
# 0.input raw data
samples_traval_hepa = pd.read_csv('Data/raw data/raw/Hepatotoxicity_1_train.csv').iloc[:, [2, 4]]  # 1597
samples_test_hepa = pd.read_csv('Data/raw data/raw/Hepatotoxicity_1_test.csv').iloc[:, [2, 5]]  # 322
samples_traval_hepa_1 = pd.read_csv('Data/raw data/raw/Hepatotoxicity_0_train.csv').iloc[:, [4, 6]] # 2889


# 1.tranform smiles to canonical smiles
mol_traval_hepa = [Chem.MolFromSmiles(x) for x in samples_traval_hepa.iloc[:, 0]]
smiles_traval_hepa = [Chem.MolToSmiles(x) for x in mol_traval_hepa]
samples_traval_hepa.iloc[:, 0] = smiles_traval_hepa

mol_traval_hepa_1 = [Chem.MolFromSmiles(x) for x in samples_traval_hepa_1.iloc[:, 0]]
smiles_traval_hepa_1 = [Chem.MolToSmiles(x) for x in mol_traval_hepa_1]
samples_traval_hepa_1.iloc[:, 0] = smiles_traval_hepa_1

mol_test_hepa = [Chem.MolFromSmiles(x) for x in samples_test_hepa.iloc[:, 0]]
smiles_test_hepa = [Chem.MolToSmiles(x) for x in mol_test_hepa]
samples_test_hepa.iloc[:, 0] = smiles_test_hepa


# 2.delete traval samples in test dataset
smiles_inter = list(set(samples_traval_hepa.iloc[:, 0]) & set(samples_test_hepa.iloc[:, 0]))
smiles_inter_1 = list(set(samples_traval_hepa_1.iloc[:, 0]) & set(samples_test_hepa.iloc[:, 0]))

samples_traval_hepa = samples_traval_hepa[~samples_traval_hepa.iloc[:, 0].isin(smiles_inter)]
samples_traval_hepa_1 = samples_traval_hepa_1[~samples_traval_hepa_1.iloc[:, 0].isin(smiles_inter_1)]

print(len(smiles_inter), len(smiles_inter_1))  # 220, 189
# Results: traval sample 1331, traval sample_1 2700, test sample 322


# 3.delete unconsistant sample to label
cols = samples_traval_hepa.columns
duplicate_smiles = samples_traval_hepa[samples_traval_hepa.duplicated(subset=[cols[0]], keep=False)]  # 94
duplicate_smiles_list = duplicate_smiles[cols[0]].unique()  # 46
indexes_to_drop = set()  # 5
smiles_to_inconsist = []  # 2
for smiles in duplicate_smiles[cols[0]].unique():
    rows = duplicate_smiles[duplicate_smiles[cols[0]] == smiles]
    if len(rows[cols[1]].unique()) > 1:
        indexes_to_drop.update(rows.index)
        smiles_to_inconsist.append(smiles)
samples_traval_hepa = samples_traval_hepa.drop(index=indexes_to_drop)  # 1326

cols_1 = samples_traval_hepa_1.columns
duplicate_smiles_1 = samples_traval_hepa_1[samples_traval_hepa_1.duplicated(subset=[cols_1[0]], keep=False)]  # 0
duplicate_smiles_list_1 = duplicate_smiles_1[cols_1[0]].unique()  # 0
indexes_to_drop_1 = set()  # 0
smiles_to_inconsist_1 = []  # 0
for smiles in duplicate_smiles_1[cols_1[0]].unique():
    rows = duplicate_smiles_1[duplicate_smiles_1[cols_1[0]] == smiles]
    if len(rows[cols_1[1]].unique()) > 1:
        indexes_to_drop_1.update(rows.index)
        smiles_to_inconsist_1.append(smiles)
samples_traval_hepa_1 = samples_traval_hepa_1.drop(index=indexes_to_drop_1)  # 2700

samples_traval_hepa_1.columns = samples_traval_hepa.columns
samples_traval_hepa_all = pd.concat([samples_traval_hepa, samples_traval_hepa_1])  # 4026
cols_all = samples_traval_hepa_all.columns
duplicate_smiles_all = samples_traval_hepa_all[samples_traval_hepa_all.duplicated(subset=[cols_all[0]], keep=False)]  # 940
duplicate_smiles_list_all = duplicate_smiles_all[cols_all[0]].unique()  # 454
indexes_to_drop_all = set()  # 99
smiles_to_inconsist_all = []  # 50
for smiles in duplicate_smiles_all[cols_all[0]].unique():
    rows = duplicate_smiles_all[duplicate_smiles_all[cols_all[0]] == smiles]
    if len(rows[cols_all[1]].unique()) > 1:
        indexes_to_drop_all.update(rows.index)
        smiles_to_inconsist_all.append(smiles)
samples_traval_hepa_all = samples_traval_hepa_all.drop(index=indexes_to_drop_all)  # 3863
# Results: traval sample 1326, traval sample_1 2700, traval sample_all 3863, test sample 322


# 4. delete duplicate samples in smiles
samples_traval_hepa = samples_traval_hepa.drop_duplicates()
smiles_unique_traval = samples_traval_hepa.iloc[:, 0].unique()  # 1281

samples_test_hepa = samples_test_hepa.drop_duplicates()
smiles_unique_test = samples_test_hepa.iloc[:, 0].unique() # 322

samples_traval_hepa_1 = samples_traval_hepa_1.drop_duplicates()
smiles_unique_1 = samples_traval_hepa_1.iloc[:, 0].unique() # 2700

samples_traval_hepa_all = samples_traval_hepa_all.drop_duplicates()
smiles_unique_all = samples_traval_hepa_all.iloc[:, 0].unique()  # 3436
# Results: traval sample 1281, traval sample_1 2700, traval sample_all 3436, test sample 322


# save samples
samples_traval_hepa.columns = ['Canonical SMILES', 'Hepatotoxicity']
samples_traval_hepa_1.columns = ['Canonical SMILES', 'Hepatotoxicity']
samples_traval_hepa_all.columns = ['Canonical SMILES', 'Hepatotoxicity']
samples_test_hepa.columns = ['Canonical SMILES', 'Hepatotoxicity']
samples_traval_hepa.to_csv('Data/raw data/Hepatotoxicity_train0.csv', index=False)
samples_traval_hepa_1.to_csv('Data/raw data/Hepatotoxicity_train1.csv', index=False)
samples_traval_hepa_all.to_csv('Data/raw data/Hepatotoxicity_train.csv', index=False)
samples_test_hepa.to_csv('Data/raw data/Hepatotoxicity_test.csv', index=False)

print('Hepatotoxicity datasets ok')

















