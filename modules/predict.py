import os
import pickle
import torch
import pandas as pd
import numpy as np
from util.util import *
from vae_generator import PlastVAEGen

#data
org_data = pd.read_pickle('../database/vae_org.pkl').to_numpy()
pl_data = pd.read_pickle('../database/vae_pl.pkl').to_numpy()

#vocab
char_weights_gdb = np.load('util/char_weights_gdb.npy')
char_weights_orgpl = np.load('util/char_weights_orgpl.npy')
with open('util/char_dict.pkl', 'rb') as f:
    char_dict = pickle.load(f)
with open('util/org_dict.pkl', 'rb') as f:
    org_dict = pickle.load(f)

#params
model_class = 'GRUGRU'
small_len = 45
big_len = 180
pretrain_params = {'MAX_LENGTH': small_len,
                   'BATCH_SIZE': 1000,
                   'KL_BETA': 0.1,
                   'MODEL_CLASS': model_class,
                   'ARCH_SIZE': 'small',
                   'CHAR_DICT': char_dict,
                   'ORG_DICT': org_dict,
                   'CHAR_WEIGHTS': char_weights_gdb}
transfer_params = {'MAX_LENGTH': big_len,
                   'BATCH_SIZE': 500,
                   'KL_BETA': 0.1,
                   'MODEL_CLASS': model_class,
                   'ARCH_SIZE': 'large',
                   'CHAR_DICT': char_dict,
                   'ORG_DICT': org_dict,
                   'CHAR_WEIGHTS': char_weights_orgpl}
predict_params = {'MAX_LENGTH': big_len,
                  'BATCH_SIZE': 500,
                  'KL_BETA': 0.1,
                  'MODEL_CLASS': model_class,
                  'ARCH_SIZE': 'large',
                  'PRED_SIZE': 'large',
                  'CHAR_DICT': char_dict,
                  'ORG_DICT': org_dict,
                  'CHAR_WEIGHTS': char_weights_orgpl,
                  'PREDICT_PROPERTY': True}

def predict_from_data(data, params, ckpt, n_splits, target_code, save_fn):

    pvg = PlastVAEGen(params=params, name='{}_prop_predictor'.format(model_class))
    pvg.load(ckpt, transfer=False, predict_property=True)

    #encode data

    smiles = list(data[:,0])
    smiles = [smi_tokenizer(smi) for smi in smiles]
    scores = data[:,1]
    encoded_smiles = np.zeros((data.shape[0], pvg.params['MAX_LENGTH']))
    for i, smi in enumerate(smiles):
        encoded_smiles[i,:] = encode_smiles(smi, pvg.params['MAX_LENGTH'], pvg.params['CHAR_DICT'])

    split_size = int(data.shape[0] / n_splits)
    os.makedirs('predictions', exist_ok=True)
    for i in range(split_size):
        if i < split_size - 1:
            reconstructions, mu, logvar, predictions = pvg.predict(encoded_smiles[i*split_size:(i+1)*split_size,:], return_all=True)
        elif i == split_size - 1:
            reconstructions, mu, logvar, predictions = pvg.predict(encoded_smiles[i*split_size:,:], return_all=True)
    np.save('predictions/{}_reconstructions_{}_{}.npy'.format(target_code, save_fn, i), reconstructions)
    np.save('predictions/{}_mu_{}_{}.npy'.format(target_code, save_fn, i), mu)
    np.save('predictions/{}_logvar_{}_{}.npy'.format(target_code, save_fn, i), logvar)
    np.save('predictions/{}_predictions_{}_{}.npy'.format(target_code, save_fn, i), predictions)

predict_from_data(org_data, predict_params, 'checkpoints/best_GRUGRU_prop_pred_epox_run2.ckpt', 10, 'org', 'epox_run2')
predict_from_data(pl_data, predict_params, 'checkpoints/best_GRUGRU_prop_pred_epox_run2.ckpt', 1, 'pl', 'epox_run2')
