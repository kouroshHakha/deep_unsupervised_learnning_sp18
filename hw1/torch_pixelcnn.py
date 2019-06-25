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
from torch.distributions import Categorical

from pixelcnn_model import PixelCNN, PixelCNNParallel

class ARPixelCNN:

    def __init__(self,
                 data_file,
                 *,
                 learning_rate=1e-3,
                 nepochs=1,
                 batch_size=128,
                 feature_size=128,
                 ):
        self.file = data_file
        self.nepochs = nepochs
        self.batch_size = batch_size
        self.feature_size = feature_size
        self.learning_rate = learning_rate

        self.xtrain: torch.Tensor = None
        self.xtest: torch.Tensor = None
        self.model = None
        self.opt = None

    def get_samples(self):
        with open(self.file, 'rb') as f:
            return pickle.load(f)

    def run_epoch(self, mode):
        s = time.time()
        self.model.train(mode == 'train')
        nsamples = self.xtrain.shape[0]
        b = self.batch_size
        nsteps = 1 if mode == 'test' else nsamples
        device = torch.device("cuda:0")
        for step in range(nsteps):
            xin = self.xtest if mode == 'test' else self.xtrain[step * b: (step + 1) * b]
            if 0 in xin.shape:
                continue
            xin = xin.to(device)
            self.model(xin.float())
            loss = self.model.loss(target=xin.long())

            if mode == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if step % 50 == 0:
                print(f'step {step} -> training loss = {loss.item()}')

        print(f'{mode} epoch average loss: {loss.item()}, finished in {time.time() - s} secs')

    def main(self, seed=10):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        data = self.get_samples()
        dtrain = cast(np.ndarray, data['train']).transpose([0,3,1,2])
        dtest = cast(np.ndarray, data['test']).transpose([0,3,1,2])

        self.xtrain = torch.from_numpy(dtrain)
        self.xtest = torch.from_numpy(dtest)

        self.model: nn.Module = PixelCNN(fm=self.feature_size)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # if torch.cuda.device_count() > 1:
            #     print("Let's use", torch.cuda.device_count(), "GPUs!")
            #     self.model = PixelCNNParallel(self.model)

        self.opt = optim.Adam(self.model.parameters())
        print('number of model parameters: ',
              sum([np.prod(p.size()) for p in self.model.parameters()]))

        for epoch in range(self.nepochs):
            s = time.time()
            print(f'epoch {epoch}')
            self.run_epoch('train')
            # self.run_epoch('test')
            print(f'training time for epoch {epoch}: {time.time() - s}')


if __name__ == '__main__':

    file = sys.argv[1]
    agent =  ARPixelCNN(file,
                        nepochs=50,
                        learning_rate=1e-3,
                        batch_size=36,
                        feature_size=128,
                        )
    agent.main()
