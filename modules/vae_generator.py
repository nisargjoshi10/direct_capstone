import os
import sys
import shutil
import imageio
import numpy as np

# torch
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# wandb
#import wandb

# me
import util.util as uu
from util.pred_blocks import ConvGRU, GRUGRU, GRUGRUPredict
from util.losses import vae_bce_loss, vae_ce_loss, vae_predictor_ce_loss

class PlastVAEGen():
    def __init__(self, params={}, name=None, weigh_freq=True, verbose=False, watch=False):
        self.params = params
        self.name = name
        self.weigh_freq = weigh_freq
        self.verbose = verbose
        self.watch = watch
        if 'LATENT_SIZE' in self.params.keys():
            self.latent_size = self.params['LATENT_SIZE']
        else:
            self.latent_size = 84
        if 'KL_BETA' not in self.params.keys():
            self.params['KL_BETA'] = 1.0
        if 'FREQ_PENALTY' not in self.params.keys():
            self.params['FREQ_PENALTY'] = 0.5
        if 'MODEL_CLASS' not in self.params.keys():
            self.params['MODEL_CLASS'] = 'ConvGRU'
        if 'ARCH_SIZE' not in self.params.keys():
            self.params['ARCH_SIZE'] = 'small'
        if 'PRED_SIZE' not in self.params.keys():
            self.params['PRED_SIZE'] = 'small'
        if 'NUM_CHAR' not in self.params.keys():
            self.params['NUM_CHAR'] = 23
        if 'EMBED_DIM' not in self.params.keys():
            self.params['EMBED_DIM'] = 30
        if 'CHAR_WEIGHTS' in self.params.keys():
            self.params['CHAR_WEIGHTS'] = torch.tensor(self.params['CHAR_WEIGHTS'], dtype=torch.float)
        else:
            self.params['CHAR_WEIGHTS'] = torch.ones(self.params['NUM_CHAR'], dtype=torch.float)
        if 'PREDICT_PROPERTY' in self.params.keys():
            self.predict_property = self.params['PREDICT_PROPERTY']
        else:
            self.predict_property = False
        self.history = {'train_loss': [],
                        'val_loss': []}
        self.best_loss = np.inf
        self.n_epochs = 0
        self.current_state = {'name': self.name,
                              'epoch': self.n_epochs,
                              'model_state_dict': None,
                              'optimizer_state_dict': None,
                              'best_loss': self.best_loss,
                              'input_shape': None,
                              'latent_size': self.latent_size,
                              'history': self.history,
                              'params': self.params}
        self.best_state = {'name': self.name,
                           'epoch': self.n_epochs,
                           'model_state_dict': None,
                           'optimizer_state_dict': None,
                           'best_loss': self.best_loss,
                           'input_shape': None,
                           'latent_size': self.latent_size,
                           'history': self.history,
                           'params': self.params}
        self.trained = False
        self.pre_trained = False

    def save(self, state, fn, path='checkpoints'):
        os.makedirs(path, exist_ok=True)
        if os.path.splitext(fn)[1] == '':
            if self.name is not None:
                fn += '_' + self.name
            fn += '.ckpt'
        else:
            if self.name is not None:
                fn, ext = fn.split('.')
                fn += '_' + self.name
                fn += '.' + ext
        torch.save(state, os.path.join(path, fn))

    def load(self, checkpoint_path, transfer=False, predict_property=False):
        loaded_checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        for key in self.current_state.keys():
            self.current_state[key] = loaded_checkpoint[key]

        if self.name is None:
            self.name = self.current_state['name']
        else:
            pass
        self.history = self.current_state['history']
        self.n_epochs = self.current_state['epoch']
        for k, v in self.current_state['params'].items():
            if k not in self.params.keys():
                self.params[k] = v
        if transfer:
            self.trained = False
            self.pre_trained = True
            self.n_pretrain_epochs = self.n_epochs
            self.n_epochs = 0
            if predict_property:
                self.predict_property = True
                if self.params['PRED_SIZE'] == 'small':
                    alpha_1 = 1 / np.sqrt(128)
                    alpha_2 = 1 / np.sqrt(128)
                    alpha_3 = 1
                    self.current_state['model_state_dict']['predictor.predict.0.weight'] = torch.FloatTensor(128, self.latent_size).uniform_(-alpha_1, alpha_1).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.0.bias'] = torch.FloatTensor(128).uniform_(-alpha_1, alpha_1).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.2.weight'] = torch.FloatTensor(128, 128).uniform_(-alpha_2, alpha_2).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.2.bias'] = torch.FloatTensor(128).uniform_(-alpha_2, alpha_2).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.4.weight'] = torch.FloatTensor(1, 128).uniform_(-alpha_3, alpha_3).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.4.bias'] = torch.FloatTensor(1).uniform_(-alpha_3, alpha_3).requires_grad_()
                elif self.params['PRED_SIZE'] == 'large':
                    alpha_1 = 1 / np.sqrt(128)
                    alpha_2 = 1 / np.sqrt(128)
                    alpha_3 = 1 / np.sqrt(64)
                    alpha_4 = 1
                    self.current_state['model_state_dict']['predictor.predict.0.weight'] = torch.FloatTensor(128, self.latent_size).uniform_(-alpha_1, alpha_1).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.0.bias'] = torch.FloatTensor(128).uniform_(-alpha_1, alpha_1).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.2.weight'] = torch.FloatTensor(128, 128).uniform_(-alpha_2, alpha_2).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.2.bias'] = torch.FloatTensor(128).uniform_(-alpha_2, alpha_2).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.4.weight'] = torch.FloatTensor(64, 128).uniform_(-alpha_3, alpha_3).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.4.bias'] = torch.FloatTensor(64).uniform_(-alpha_3, alpha_3).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.6.weight'] = torch.FloatTensor(1, 64).uniform_(-alpha_4, alpha_4).requires_grad_()
                    self.current_state['model_state_dict']['predictor.predict.6.bias'] = torch.FloatTensor(1).uniform_(-alpha_4, alpha_4).requires_grad_()
                    pass
            else:
                self.predict_property = False
        else:
            self.trained = True
            self.pre_trained = False
            if predict_property:
                self.predict_property = True
                if self.params['MODEL_CLASS'] == 'GRUGRU':
                    self.network = GRUGRUPredict(self.current_state['input_shape'], self.current_state['latent_size'], embed_dim=self.params['EMBED_DIM'], arch_size=self.params['ARCH_SIZE'], pred_size=self.params['PRED_SIZE'])
            else:
                if self.params['MODEL_CLASS'] == 'ConvGRU':
                    self.network = ConvGRU(self.current_state['input_shape'], self.current_state['latent_size'])
                elif self.params['MODEL_CLASS'] == 'ConvbiGRU':
                    self.network = ConvGRU(self.current_state['input_shape'], self.current_state['latent_size'], dec_bi=True)
                elif self.params['MODEL_CLASS'] == 'GRUGRU':
                    self.network = GRUGRU(self.current_state['input_shape'], self.current_state['latent_size'], embed_dim=self.params['EMBED_DIM'], arch_size=self.params['ARCH_SIZE'])
                elif self.params['MODEL_CLASS'] == 'biGRUGRU':
                    self.network = GRUGRU(self.current_state['input_shape'], self.current_state['latent_size'], embed_dim=self.params['EMBED_DIM'], enc_bi=True, arch_size=self.params['ARCH_SIZE'])
                elif self.params['MODEL_CLASS'] == 'biGRUbiGRU':
                    self.network = GRUGRU(self.current_state['input_shape'], self.current_state['latent_size'], embed_dim=self.params['EMBED_DIM'], enc_bi=True, dec_bi=True, arch_size=self.params['ARCH_SIZE'])
                self.predict_property = False
            self.network.load_state_dict(self.current_state['model_state_dict'])

    def initiate(self, data):
        """
        This function not only loads data, but also builds the network
        architecture. This must be done after the data is loaded because
        the number of input channels is dependent on the max length of smiles
        strings and number of unique characters.
        """
        # Setting up parameters
        self.all_smiles = data[:,0]
        self.all_lls = data[:,1]
        del data
        self.all_smiles = [uu.smi_tokenizer(smi) for smi in self.all_smiles]
        if 'MAX_LENGTH' not in self.params.keys():
            self.params['MAX_LENGTH'] = 100
        if 'TRAIN_SPLIT' not in self.params.keys():
            self.params['TRAIN_SPLIT'] = 0.8

        # One-hot encoding smiles below the max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.params['MAX_LENGTH']]
        del self.all_smiles, self.all_lls
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        del self.usable_data
        # self.params['CHAR_DICT'], self.params['ORD_DICT'] = uu.get_smiles_vocab(self.usable_smiles)
        self.params['NUM_CHAR'] = len(self.params['CHAR_DICT'])
        self.params['PAD_NUM'] = self.params['CHAR_DICT']['_']
        self.encoded = torch.empty((len(self.usable_smiles), self.params['MAX_LENGTH']))
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:] = torch.tensor(uu.encode_smiles(sm, self.params['MAX_LENGTH'], self.params['CHAR_DICT']))
        del self.usable_smiles
        self.input_shape = (self.params['NUM_CHAR'], self.params['MAX_LENGTH'])

        # Data preparation
        self.params['N_SAMPLES'] = self.encoded.shape[0]
        self.params['N_TRAIN'] = int(self.params['N_SAMPLES'] * self.params['TRAIN_SPLIT'])
        self.params['N_TEST'] = self.params['N_SAMPLES'] - self.params['N_TRAIN']
        self.rand_idxs = np.random.choice(np.arange(self.params['N_SAMPLES']), size=self.params['N_SAMPLES'])
        self.params['TRAIN_IDXS'] = self.rand_idxs[:self.params['N_TRAIN']]
        self.params['VAL_IDXS'] = self.rand_idxs[self.params['N_TRAIN']:]
        # if self.weigh_freq:
        #     self.params['CHAR_WEIGHTS'] = torch.tensor(uu.get_char_weights(np.array(self.usable_smiles)[self.params['TRAIN_IDXS']], self.params, self.params['FREQ_PENALTY']), dtype=torch.float)
        # else:
        #     self.params['CHAR_WEIGHTS'] = torch.ones(self.params['NUM_CHAR'], dtype=torch.float)

        X_train = self.encoded[self.params['TRAIN_IDXS'],:]
        X_val = self.encoded[self.params['VAL_IDXS'],:]
        y_train = self.usable_lls[self.params['TRAIN_IDXS']]
        y_val = self.usable_lls[self.params['VAL_IDXS']]
        self.train_data = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
        self.val_data = np.concatenate([X_val, y_val.reshape(-1,1)], axis=1)

        # Build network
        if self.trained:
            assert self.input_shape == self.current_state['input_shape'], "ERROR - Shape of data different than that used to train loaded model"
            assert self.latent_size == self.current_state['latent_size'], "ERROR - Latent space of trained model unequal to input parameter"
        else:
            if self.predict_property:
                if self.params['MODEL_CLASS'] == 'GRUGRU':
                    self.network = GRUGRUPredict(self.input_shape, self.latent_size, embed_dim=self.params['EMBED_DIM'], arch_size=self.params['ARCH_SIZE'], pred_size=self.params['PRED_SIZE'])
            else:
                if self.params['MODEL_CLASS'] == 'ConvGRU':
                    self.network = ConvGRU(self.input_shape, self.latent_size)
                elif self.params['MODEL_CLASS'] == 'ConvbiGRU':
                    self.network = ConvGRU(self.input_shape, self.latent_size, dec_bi=True)
                elif self.params['MODEL_CLASS'] == 'GRUGRU':
                    self.network = GRUGRU(self.input_shape, self.latent_size, embed_dim=self.params['EMBED_DIM'], arch_size=self.params['ARCH_SIZE'])
                elif self.params['MODEL_CLASS'] == 'biGRUGRU':
                    self.network = GRUGRU(self.input_shape, self.latent_size, embed_dim=self.params['EMBED_DIM'], enc_bi=True, arch_size=self.params['ARCH_SIZE'])
                elif self.params['MODEL_CLASS'] == 'biGRUbiGRU':
                    self.network = GRUGRU(self.input_shape, self.latent_size, embed_dim=self.params['EMBED_DIM'], enc_bi=True, dec_bi=True, arch_size=self.params['ARCH_SIZE'])
        if self.pre_trained:
            self.network.load_state_dict(self.current_state['model_state_dict'])

        # Update state dictionaries
        self.current_state['input_shape'] = self.input_shape
        self.best_state['input_shape'] = self.input_shape
        self.current_state['latent_size'] = self.latent_size
        self.best_state['latent_size'] = self.latent_size

        if self.watch:
            pass
            #wandb.init(project='plastgenvae')

    def trained_initiate(self, data):
        """
        Function analogous to `self.initiate` except some parameters do not need to be
        re-initialized because model has already been trained for some number of
        epochs (you must use the same data that model was initially trained on)
        """
        # Setting up parameters
        self.all_smiles = data[:,0]
        self.all_lls = data[:,1]
        self.all_smiles = [uu.smi_tokenizer(smi) for smi in self.all_smiles]

        # One-hot encoding smiles below the max length
        self.usable_data = [(ll, sm) for ll, sm in zip(self.all_lls, self.all_smiles) if len(sm) < self.params['MAX_LENGTH']]
        self.usable_lls = np.array([x[0] for x in self.usable_data])
        self.usable_smiles = [x[1] for x in self.usable_data]
        self.encoded = torch.empty((len(self.usable_smiles), self.params['MAX_LENGTH']))
        for i, sm in enumerate(self.usable_smiles):
            self.encoded[i,:] = torch.tensor(uu.encode_smiles(sm, self.params['MAX_LENGTH'], self.params['CHAR_DICT']))
        self.input_shape = (self.params['NUM_CHAR'], self.params['MAX_LENGTH'])
        self.current_state['input_shape'] = self.input_shape
        self.best_state['input_shape'] = self.input_shape

        # Data preparation
        X_train = self.encoded[self.params['TRAIN_IDXS'],:]
        X_val = self.encoded[self.params['VAL_IDXS'],:]
        y_train = self.usable_lls[self.params['TRAIN_IDXS']]
        y_val = self.usable_lls[self.params['VAL_IDXS']]
        self.train_data = np.concatenate([X_train, y_train.reshape(-1,1)], axis=1)
        self.val_data = np.concatenate([X_val, y_val.reshape(-1,1)], axis=1)

        if self.watch:
            pass
            #wandb.init(project='plastgenvae')

    def train(self, data, epochs=100, save_last=True, save_best=True, save_latest=True, log=True, log_latent=False, make_grad_gif=False):
        """
        Function to train model with loaded data
        """
        # Setting up parameters
        if 'BATCH_SIZE' in self.params.keys():
            self.batch_size = self.params['BATCH_SIZE']
        else:
            self.batch_size = 1000
        if 'LEARNING_RATE' in self.params.keys():
            self.lr = self.params['LEARNING_RATE']
        else:
            self.lr = 1e-4

        if not self.trained:
            self.initiate(data)
        elif self.trained:
            self.trained_initiate(data)

        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.network.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()
        if make_grad_gif:
            os.mkdir('gif')
            images = []
            frame = 0

        # Save constant params to state dicts
        if save_best:
            self.best_state['params'] = self.params
        if save_last:
            self.current_state['params'] = self.params

        # Create data iterables
        train_loader = torch.utils.data.DataLoader(self.train_data,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=0,
                                                   pin_memory=False,
                                                   drop_last=True)
        val_loader = torch.utils.data.DataLoader(self.val_data,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 num_workers=0,
                                                 pin_memory=False,
                                                 drop_last=True)


        # Create optimizer
        if self.trained:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            self.optimizer.load_state_dict(self.current_state['optimizer_state_dict'])
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)

        # Create loss function
        if self.predict_property:
            loss_func = vae_predictor_ce_loss
        else:
            loss_func = vae_ce_loss

        # Set up logger
        if log and not self.trained:
            os.makedirs('trials', exist_ok=True)
            if self.name is not None:
                log_file = open('trials/log{}.txt'.format('_'+self.name), 'a')
            else:
                log_file = open('trials/log.txt', 'a')
            log_file.write('epoch,batch_idx,data_type,tot_loss,bce_loss,kld_loss,pred_loss\n')
            log_file.close()

        if log_latent:
            os.makedirs('latent_arrays', exist_ok=True)
            self.latent_mu_train = np.zeros((epochs, self.train_data.shape[0], self.latent_size+1))
            self.latent_mu_val = np.zeros((epochs, self.val_data.shape[0], self.latent_size+1))

        # Set up metric evaluation
        if self.watch:
            pass
            #wandb.watch(self.network)

        # Epoch Looper
        for i, epoch in enumerate(range(epochs)):
            if self.verbose:
                epoch_counter = '[{}/{}]'.format(epoch+1, epochs)
                progress_bar = '['+'-'*50+']'
                sys.stdout.write("\r\033[K"+epoch_counter+' '+progress_bar)

            # Train Loop
            self.network.train()
            h = self.network.decoder.init_hidden(self.params['BATCH_SIZE']).data
            losses = []
            for batch_idx, data in enumerate(train_loader):
                self.network.zero_grad()
                if use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data[:,:-1])
                targets = torch.autograd.Variable(data[:,-1])
                targets = targets.float()
                x_decode, mu, logvar, predictions = self.network(x)
                loss, bce, kld, mse = loss_func(x, x_decode, mu, logvar, targets, predictions, self.params['CHAR_WEIGHTS'], self.params['KL_BETA'])
                if self.watch:
                    pass
                    #wandb.log({'BCE Loss': bce.item(), 'KLD Loss': kld.item()})

                loss.backward()
                if make_grad_gif:
                    plt = uu.plot_grad_flow(self.network.named_parameters())
                    plt.title('Epoch {}  Frame {}'.format(epoch, frame))
                    fn = 'gif/{}.png'.format(frame)
                    plt.savefig(fn)
                    plt.close()
                    images.append(imageio.imread(fn))
                    frame += 1
                self.optimizer.step()

                losses.append(loss.item())
                if log:
                    if self.name is not None:
                        log_file = open('trials/log{}.txt'.format('_'+self.name), 'a')
                    else:
                        log_file = open('trials/log.txt', 'a')
                    log_file.write('{},{},{},{},{},{},{}\n'.format(self.n_epochs,batch_idx,'train',loss.item(),bce.item(),kld.item(),mse.item()))
                    log_file.close()
                if log_latent:
                    self.latent_mu_train[i,batch_idx*self.batch_size:(batch_idx+1)*self.batch_size,:-1] = mu.data.cpu().numpy()
                    self.latent_mu_train[i,batch_idx*self.batch_size:(batch_idx+1)*self.batch_size,-1] = targets.data.cpu().numpy()
            train_loss = np.mean(losses)
            self.history['train_loss'].append(train_loss)

            # Val Loop
            self.network.eval()
            h = self.network.decoder.init_hidden(self.params['BATCH_SIZE']).data
            losses = []
            for batch_idx, data in enumerate(val_loader):
                if use_gpu:
                    data = data.cuda()

                x = torch.autograd.Variable(data[:,:-1])
                targets = torch.autograd.Variable(data[:,-1])
                targets = targets.float()
                x_decode, mu, logvar, predictions = self.network(x)
                loss, bce, kld, mse = loss_func(x, x_decode, mu, logvar, targets, predictions, self.params['CHAR_WEIGHTS'], self.params['KL_BETA'])
                losses.append(loss.item())
                if log:
                    if self.name is not None:
                        log_file = open('trials/log{}.txt'.format('_'+self.name), 'a')
                    else:
                        log_file = open('trials/log.txt', 'a')
                    log_file.write('{},{},{},{},{},{},{}\n'.format(self.n_epochs,batch_idx,'test',loss.item(),bce.item(),kld.item(),mse.item()))
                    log_file.close()
                if log_latent:
                    self.latent_mu_val[i,batch_idx*self.batch_size:(batch_idx+1)*self.batch_size,:-1] = mu.data.cpu().numpy()
                    self.latent_mu_val[i,batch_idx*self.batch_size:(batch_idx+1)*self.batch_size,-1] = targets.data.cpu().numpy()
            val_loss = np.mean(losses)
            self.history['val_loss'].append(val_loss)
            print('Epoch - {}  Train Loss - {}  Val Loss - {}'.format(self.n_epochs,
                                                                      round(train_loss, 2),
                                                                      round(val_loss, 2)))
            self.n_epochs += 1

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                if save_best:
                    self.best_state['epoch'] = self.n_epochs
                    self.best_state['model_state_dict'] = self.network.state_dict()
                    self.best_state['optimizer_state_dict'] = self.optimizer.state_dict()
                    self.best_state['best_loss'] = self.best_loss
                    self.best_state['history'] = self.history
                    self.save(self.best_state, 'best')
            self.current_state['epoch'] = self.n_epochs
            self.current_state['model_state_dict'] = self.network.state_dict()
            self.current_state['optimizer_state_dict'] = self.optimizer.state_dict()
            self.current_state['best_loss'] = self.best_loss
            self.current_state['history'] = self.history
            self.best_state['history'] = self.history

            if save_latest:
                self.save(self.current_state, 'latest')

            if log_latent:
                np.save(os.path.join('latent_arrays', 'mu_train_{}'.format(self.name)), self.latent_mu_train)
                np.save(os.path.join('latent_arrays', 'mu_val_{}'.format(self.name)), self.latent_mu_val)
        if save_last:
            self.save(self.current_state, 'latest')
            self.save(self.best_state, 'best')
        self.trained = True
        if make_grad_gif:
            imageio.mimsave('grads4.gif', images)
            shutil.rmtree('gif')

    def predict(self, data, return_all=False):
        """
        Predicts output given a set of input data (model must already be trained)
        """
        data = torch.from_numpy(data)
        torch.backends.cudnn.benchmark = True
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.network.cuda()
            self.params['CHAR_WEIGHTS'] = self.params['CHAR_WEIGHTS'].cuda()
            data = data.cuda()
        self.network.eval()
        h = self.network.decoder.init_hidden(data.shape[0])
        x = torch.autograd.Variable(data)
        x_decode, mu, logvar, predictions = self.network(x)
        if return_all:
            return x_decode.cpu().detach().numpy(), mu.cpu().detach().numpy(), logvar.cpu().detach().numpy(), predictions.cpu().detach().numpy()
        else:
            return x_decode.cpu().detach().numpy()
