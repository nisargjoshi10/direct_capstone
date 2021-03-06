import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def vae_ce_loss(x, x_decode, mu, logvar, targets, predictions, weights, beta=1):
    x = x.long()
    x_decode = x_decode.permute(0, 2, 1)
    x = x.contiguous().view(-1)
    x_decode = x_decode.contiguous().view(-1, x_decode.size(2))
    BCE = F.cross_entropy(x_decode, x, reduction='mean', weight=weights)
    # BCE = F.nll_loss(x_decode, argmax, reduction='mean')
    # BCE = max_len * F.binary_cross_entropy(x_decode, x, reduction='mean')
    # BCE = F.binary_cross_entropy_with_logits(x_decode, x, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    MSE = predictions.mean()
    return BCE + KLD, BCE, KLD, MSE

def vae_predictor_ce_loss(x, x_decode, mu, logvar, targets, predictions, weights, beta=1):
    x = x.long()
    x_decode = x_decode.permute(0, 2, 1)
    x = x.contiguous().view(-1)
    x_decode = x_decode.contiguous().view(-1, x_decode.size(2))
    targets = targets.view(-1, 1)
    BCE = F.cross_entropy(x_decode, x, reduction='mean', weight=weights)
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    #logfile = open('trials/mse_tracker.txt', 'a')
    #for i in range(targets.shape[0]):
    #    logfile.write(str(round(targets[i].item(), 3))+','+str(round(predictions[i].item(), 3))+'\n')
    #logfile.close()
    MSE = F.mse_loss(targets, predictions, reduction='mean')
    return BCE + KLD + MSE, BCE, KLD, MSE

def vae_bce_loss(x, x_decode, mu, logvar, max_len, beta=1):
    x = x.permute(0, 2, 1)
    x_decode = x_decode.permute(0, 2, 1)
    x = x.contiguous().view(-1, x.size(2))
    x_decode = x_decode.contiguous().view(-1, x_decode.size(2))
    BCE = F.binary_cross_entropy_with_logits(x_decode, x, reduction='mean')
    KLD = beta * -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

def ce_loss(x, x_decode):
    x = x.permute(0, 2, 1)
    x_decode = x_decode.permute(0, 2, 1)
    x = x.contiguous().view(-1, x.size(2))
    x_decode = x_decode.contiguous().view(-1, x_decode.size(2))
    _, argmax = torch.max(x, 1)
    BCE = F.cross_entropy(x_decode, argmax, reduction='mean')
    return BCE

def bce_loss(x, x_decode):
    x = x.permute(0, 2, 1)
    x_decode = x_decode.permute(0, 2, 1)
    x = x.contiguous().view(-1, x.size(2))
    x_decode = x_decode.contiguous().view(-1, x_decode.size(2))
    BCE = F.binary_cross_entropy_with_logits(x_decode, x, reduction='mean')
    return BCE
