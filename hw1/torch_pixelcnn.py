from typing import cast

import pickle
import sys
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from pixelcnn_model import PixelCNN

from logger import TorchLogger

class ARPixelCNN:

    def __init__(self, data_file, save_dir, **kwargs):

        self.file = data_file
        self.nepochs = kwargs.get('nepochs', 1)
        self.batch_size = kwargs.get('batch_size', 128)
        self.feature_size = kwargs.get('feature_size', 128)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)

        self.xtrain: torch.Tensor = None
        self.xtest: torch.Tensor = None
        self.model = None
        self.opt = None

        self.logger = TorchLogger(save_dir, meta_data=kwargs)

    def get_samples(self):
        with open(self.file, 'rb') as f:
            return pickle.load(f)

    def run_epoch(self, mode, data, device):
        s = time.time()

        self.model.to(device)
        self.model.train(mode == 'train')

        nsamples = data.shape[0]
        b = self.batch_size
        nsteps = 1 if mode == 'test' else nsamples // b
        epoch_loss = 0
        for step in range(nsteps):
            xin = data[step * b: (step + 1) * b]
            if 0 in xin.shape:
                continue
            xin = xin.float().to(device)

            if mode == 'test':
                with torch.no_grad():
                    self.model(xin)
            else:
                self.model(xin)

            loss = self.model.loss(target=xin.long())
            epoch_loss += loss.item() / nsteps

            if mode == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if step % 50 == 0 :
                    print(f'step {step} -> training loss = {loss.item()}')

        print(f'{mode} epoch average loss: {epoch_loss}, finished in {time.time() - s} secs')

    def main(self, seed=10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data = self.get_samples()
        dtrain = cast(np.ndarray, data['train']).transpose([0,3,1,2])
        dtest = cast(np.ndarray, data['test']).transpose([0,3,1,2])

        self.xtrain = torch.from_numpy(dtrain)
        self.xtest = torch.from_numpy(dtest)

        device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

        self.model: nn.Module = PixelCNN(fm=self.feature_size)
        self.model = self.model.to(device)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     self.model = PixelCNNParallel(self.model)

        self.opt = optim.Adam(self.model.parameters())
        nparams = sum([np.prod(p.size()) for p in self.model.parameters()])
        self.logger.log(f'number of model parameters: {nparams}')

        for epoch in range(self.nepochs):
            s = time.time()
            self.logger.log(f'epoch {epoch}')
            self.run_epoch('train', self.xtrain, device)
            self.logger.log(f'training memory: '
                            f'{torch.cuda.max_memory_allocated() / 1024 ** 3:10.2f} GB', show=False)
            torch.cuda.reset_max_memory_allocated()
            self.run_epoch('test', self.xtest, device)
            self.logger.log(f'test memory: '
                            f'{torch.cuda.max_memory_allocated() / 1024 ** 3:10.2f} GB', show=False)

            self.logger.log(f'training time for epoch {epoch}: {time.time() - s}')

            # Saving the model
            print("Saving Checkpoint!")
            self.logger.save_model(self.model)
            print('Checkpoint Saved')


if __name__ == '__main__':

    file = sys.argv[1]
    save_dir = sys.argv[2]
    agent =  ARPixelCNN(file,
                        save_dir,
                        nepochs=50,
                        learning_rate=1e-3,
                        batch_size=128,
                        feature_size=128)
    agent.main()
