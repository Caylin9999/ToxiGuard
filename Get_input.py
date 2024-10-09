import pandas as pd
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML import Descriptors
from sklearn.preprocessing import StandardScaler
from rdkit.ML.Descriptors import MoleculeDescriptors
print(rdkit.__version__)
from sklearn.model_selection import train_test_split

# label_lists = ['Nephrotoxicity', 'Respiratory Toxicity', 'Hepatotoxicity', 'Cardiotoxicity']
label_lists = ['Cardiotoxicity']

def descriptors_generate(smiles):
    mol_list = [Chem.MolFromSmiles(x) for x in smiles]
    des_list = pd.read_csv('Data/raw data/descriptors_list.csv').iloc[:, 1].tolist()
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(des_list)
    data_desc = [calculator.CalcDescriptors(mol) for mol in mol_list]
    df_desc = pd.DataFrame(data_desc, columns=des_list)
    # df_desc.insert(0, 'Canonical SMILES', smiles.tolist())
    return df_desc


def FCFP_generate(smiles):
    d = 6
    dim = 2048
    mol_list = [Chem.MolFromSmiles(x) for x in smiles]
    fp_list = [AllChem.GetMorganFingerprintAsBitVect(x, int(d / 2), nBits=dim, useFeatures=True) for x in mol_list]
    fp_data = []
    for i in range(len(fp_list)):
        fp_arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(fp_list[i], fp_arr)
        fp_data.append(fp_arr)
    fp_df = pd.DataFrame(fp_data)
    # fp_df.insert(0, 'Canonical SMILES', smiles.tolist())
    return fp_df


for label_name in label_lists:
    samples_train = pd.read_csv('Data/raw data/' + str(label_name) + '_train.csv')
    df_desc_train = descriptors_generate(samples_train.iloc[:, 0])
    fp_df_train = FCFP_generate(samples_train.iloc[:, 0])
    samples_test = pd.read_csv('Data/raw data/' + str(label_name) + '_test.csv')
    df_desc_test = descriptors_generate(samples_test.iloc[:, 0])
    fp_df_test = FCFP_generate(samples_test.iloc[:, 0])
    samples_train_raw = pd.concat([samples_train, df_desc_train, fp_df_train], axis=1)
    samples_test_raw = pd.concat([samples_test, df_desc_test, fp_df_test], axis=1)
    samples_train = samples_train_raw.dropna()  # drop 1
    samples_test = samples_test_raw.dropna()

    # scaler = StandardScaler()
    # scaler.fit(samples_train.iloc[:, 3:212])
    # data_train = scaler.transform(samples_train.iloc[:, 3:212])
    # data_test = scaler.transform(samples_test.iloc[:, 3:212])
    # samples_train.iloc[:, 3:212] = pd.DataFrame(data_train, columns=list(samples_train.columns)[3:212])
    # samples_test.iloc[:, 3:212] = pd.DataFrame(data_test, columns=list(samples_test.columns)[3:212])
    # samples_train = samples_train.fillna(0)
    # samples_test = samples_test.fillna(0)

    samples_train.to_csv('Data/train_' + str(label_name) + '.csv')
    samples_test.to_csv('Data/test_' + str(label_name) + '.csv')
    print(label_name)


