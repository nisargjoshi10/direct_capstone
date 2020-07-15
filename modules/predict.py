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

def predict_from_data(data, params, ckpt, n_splits, save_fn):

    pvg = PlastVAEGen(params=params, name='{}_prop_predictor'.format(model_class))
    pvg.load(ckpt, transfer=True, predict_property=True)

    #encode data

    smiles = list(data[:,0])
    smiles = [smi_tokenizer(smi) for smi in smiles]
    scores = data[:,1]
    encoded_smiles = np.zeros((data.shape[0], pvg.params['MAX_LENGTH']))
    for i, smi in enumerate(smiles):
        encoded_smiles[i,:] = encode_smiles(smi, pvg.params['MAX_LENGTH'], pvg.params['CHAR_DICT'])

    split_size = int(data.shape[0] / n_splits)
    for i in range(split_size):
        if i < split_size - 1:
            reconstructions, mu, logvar, predictions = pvg.predict(encoded_smiles[i*split_size:(i+1)*split_size,:], return_all=True)
            print(reconstructions.shape, mu.shape, logvar.shape, predictions.shape)
        elif i == split_size - 1:
            reconstructions, mu, logvar, predictions = pvg.predict(encoded_smiles[i*split_size:,:], return_all=True)
            print(reconstructions.shape, mu.shape, logvar.shape, predictions.shape)
    # for i in range(10):
    #     if i < 9:
    #         org_reconstructions, org_mu, org_logvar, org_predictions = pvg.predict(org_encoded_smiles[i*5000:(i+1)*5000,:], return_all=True)
    #     elif i == 9:
    #         org_reconstructions, org_mu, org_logvar, org_predictions = pvg.predict(org_encoded_smiles[i*5000:,:], return_all=True)
    #     #pl_reconstructions, pl_mu, pl_logvar, pl_predictions = pvg.predict(pl_encoded_smiles, return_all=True)
    #
    #     os.makedirs('predictions', exist_ok=True)
    #     np.save('predictions/org_reconstructions_{}.npy'.format(i), org_reconstructions)
    #     np.save('predictions/org_mu_{}.npy'.format(i), org_mu)
    #     np.save('predictions/org_logvar_{}.npy'.format(i), org_logvar)
    #     np.save('predictions/org_predictions_{}.npy'.format(i), org_predictions)
    #     #np.save('predictions/pl_reconstructions.npy', pl_reconstructions)
    #     #np.save('predictions/pl_mu.npy', pl_mu)
    #     #np.save('predictions/pl_logvar.npy', pl_logvar)
    #     #np.save('predictions/pl_predictions.npy', pl_predictions)

predict_from_data(org_data, predict_params, 'checkpoints/latest_GRUGRU_prop_pred_epox_run2.ckpt', 10, 'org')
