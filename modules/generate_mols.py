import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.util import *
from vae_generator import PlastVAEGen

### Load model parameters
char_weights_gdb = np.load('util/char_weights_gdb.npy')
char_weights_orgpl = np.load('util/char_weights_orgpl.npy')
with open('util/char_dict.pkl', 'rb') as f:
    char_dict = pickle.load(f)
with open('util/org_dict.pkl', 'rb') as f:
    org_dict = pickle.load(f)

### Load model
pvg = PlastVAEGen(params={'MAX_LENGTH': 180,
                          'BATCH_SIZE': 10,
                          'MODEL_CLASS': 'GRUGRU',
                          'ARCH_SIZE': 'large',
                          'CHAR_DICT': char_dict,
                          'ORG_DICT': org_dict,
                          'CHAR_WEIGHTS': char_weights_orgpl},
                          name='test')
pvg.load('checkpoints/latest_GRUGRU_prop_pred_1mil.ckpt', transfer=False, predict_property=True)

### Load data
org_data = pd.read_pickle('../database/vae_org.pkl').to_numpy()
org_smiles = list(org_data[:,0])
org_smiles = np.array([smi_tokenizer(smi) for smi in org_smiles])
org_ll = org_data[:,1]
org_encoded_smiles = np.zeros((org_data.shape[0], 180))
for i, sm in enumerate(org_smiles):
    org_encoded_smiles[i,:] = encode_smiles(sm, 180, pvg.params['CHAR_DICT'])

pl_data = pd.read_pickle('../database/vae_pl.pkl').to_numpy()
pl_smiles = list(pl_data[:,0])
pl_smiles = np.array([smi_tokenizer(smi) for smi in pl_smiles])
pl_ll = pl_data[:,1]
pl_encoded_smiles = np.zeros((pl_data.shape[0], 180))
for i, sm in enumerate(pl_smiles):
    pl_encoded_smiles[i,:] = encode_smiles(sm, 180, pvg.params['CHAR_DICT'])

### Load model predictions
org_mu = np.load('run_data/predictions/org_mu.npy')
org_logvar = np.load('run_data/predictions/org_logvar.npy')

pl_mu = np.load('run_data/predictions/pl_mu.npy')
pl_logvar = np.load('run_data/predictions/pl_logvar.npy')

### Sort indices by plasticizer likelihood
pl_sort = np.argsort(pl_ll)[::-1]
org_sort = np.argsort(org_ll)[::-1]
org_sort_top = org_sort[:100]

n_iter = 50
temp = 0.5
n_interp_ps = 9
candidate_smiles = list(np.load('candidate_smiles_w_org.npy', allow_pickle=True))
for i in range(pl_sort.shape[0]):
    if i == 0:
        pass
    else:
        for j in range(org_sort_top.shape[0]):
            if j >= i:
                print('{}_{}'.format(i,j), len(candidate_smiles))
                idx_1 = i
                idx_2 = j
                mu_1 = pl_mu[pl_sort[i]]
                mu_2 = org_mu[org_sort_top[j]]
                logvar_1 = pl_logvar[pl_sort[i]]
                logvar_2 = org_logvar[org_sort_top[j]]
                z_1 = pvg.network.encoder.reparameterize(torch.tensor(mu_1), torch.tensor(logvar_1)).numpy()
                z_2 = pvg.network.encoder.reparameterize(torch.tensor(mu_2), torch.tensor(logvar_2)).numpy()

                interp_intervals = np.linspace(0, 1, n_interp_ps+2)[1:-1]
                for interval in interp_intervals:
                    interp_p = slerp(z_1, z_2, interval)
                    for _ in range(n_iter):
                        interp_decode = decode_z(interp_p, pvg, temp=temp)
                        candidate = Chem.MolFromSmiles(interp_decode)
                        if candidate is not None and interp_decode not in candidate_smiles and interp_decode not in pl_smiles and interp_decode not in org_smiles:
                            candidate_smiles.append(interp_decode)
                candidate_save = np.array(candidate_smiles)
                np.save('candidate_smiles_w_org.npy', candidate_save)
