
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import numpy as np

from utils.data import divide_ds
from utils.logger import TorchLogger

from realnvp2d import realNVP
import yaml
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from scipy.interpolate import griddata
import pdb

torch.random.manual_seed(10)
np.random.seed(10)

def clip_grad_norm(optimizer, max_norm, norm_type=2):
    """Clip the norm of the gradients for all parameters under `optimizer`.

    Args:
        optimizer (torch.optim.Optimizer):
        max_norm (float): The maximum allowable norm of gradients.
        norm_type (int): The type of norm to use in computing gradient norms.
    """
    for group in optimizer.param_groups:
        utils.clip_grad_norm_(group['params'], max_norm, norm_type)

def sample_data():
    count = 10000
    rand = np.random.RandomState(0)
    a: np.ndarray = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b: np.ndarray = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c: np.ndarray = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
                          -np.sin(np.linspace(0, np.pi, count // 3))]

    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]


class Agent:

    def __init__(self,
                 nepochs=10,
                 batch_size=32,
                 hidden_layers=(40, 40, 40),
                 n_layers = 4,
                 loaded = False,
                 ):

        self.batch_size = batch_size
        self.hidden_layers = list(hidden_layers)
        self.n_layers = n_layers
        self.nepochs = nepochs

        self.optimizer = None
        self.model = None

        meta_data = dict(
            nepochs=nepochs,
            batch_size=batch_size,
            hidden_layers=hidden_layers,
            n_layers=n_layers,
        )
        if not loaded:
            self.logger = TorchLogger('data', meta_data)

    @classmethod
    def from_meta_file(cls, file):
        with open(file, 'r') as f:
            meta_data = yaml.load(f)
        return Agent(nepochs=1, **meta_data, loaded=True)

    def run_epoch(self, data, device, mode='train'):
        self.model.train(mode == 'train')
        ndata, nin = data.shape
        b = self.batch_size
        nsteps = ndata // b
        for step in range(nsteps):
            xbatch = data[step * b: (step + 1) * b].to(device)
            z, ll = self.model(xbatch)
            nll = -torch.mean(ll, dim=-1)
            if mode == 'train':
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_norm(self.optimizer, max_norm=100)
                self.optimizer.step()

            if step % 100 == 0 and mode == 'train':
                self.logger.print(f'[train] neg_ll = {nll}')

        self.logger.log(f'[{mode}] epoch finished nll = {nll}', show=True)

    def run_main(self):
        data_x, data_y = sample_data()

        train_x, test_x, train_y, test_y = divide_ds(data_x, data_y, 0.8)
        train_x = torch.from_numpy(train_x).float()
        test_x = torch.from_numpy(test_x).float()
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model: nn.Module = realNVP(self.hidden_layers, self.n_layers)
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters())

        self.model.train(True)
        for epoch in range(self.nepochs):
            self.logger.log(f'epoch = {epoch}')
            self.run_epoch(train_x, device, mode='train')
            self.run_epoch(test_x, device, mode='test')
            self.logger.save_model(self.model)

    def show_model_density(self, xrange, num=16, checkpoint_fname=None):

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model: nn.Module = realNVP(self.hidden_layers, self.n_layers)
        self.model.load_state_dict(torch.load(checkpoint_fname, map_location=device))
        self.model.to(device)
        self.model.eval()

        x = np.linspace(xrange[0], xrange[1], num)
        x0, x1 = np.meshgrid(x, x)
        data_x = np.array([[x0[i, j], x1[i, j]] for i in range(num) for j in range(num)])
        x_array = torch.from_numpy(data_x).float()
        z, y_arr = self.model(x_array.to(device))
        y_arr = torch.detach(y_arr.to('cpu')).numpy()
        y_arr = np.exp(y_arr)
        y_arr = griddata(data_x, y_arr, (x0, x1), method='nearest')
        fig = plt.figure(2)
        ax = fig.gca()
        ax.imshow(y_arr, interpolation='gaussian', origin='low',
                  extent=[x0.min(), x0.max(), x1.min(), x1.max()])

        plt.show()

    def show_model_latent(self):

        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model: nn.Module = realNVP(self.hidden_layers, self.n_layers)
        self.model.load_state_dict(torch.load(checkpoint_fname, map_location=device))
        self.model.to(device)
        self.model.eval()

        data_x, data_y = sample_data()
        x_array = torch.from_numpy(data_x).float()
        z, _ = self.model(x_array.to(device))
        z = torch.detach(z.to('cpu')).numpy()
        plt.scatter(z[:,0][data_y == 0], z[:,1][data_y == 0], c='r', s=1, label='0')
        plt.scatter(z[:,0][data_y == 1], z[:,1][data_y == 1], c='b', s=1, label='1')
        plt.scatter(z[:,0][data_y == 2], z[:,1][data_y == 2], c='g', s=1, label='2')
        plt.legend()

        plt.show()
        pdb.set_trace()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        checkpoint_fname = Path(sys.argv[1])
        meta_fname = checkpoint_fname.parent / 'meta.yaml'
        agent = Agent.from_meta_file(meta_fname)
        # agent.show_model_density(xrange=[-4, 4], num=100, checkpoint_fname=checkpoint_fname)
        agent.show_model_latent()
    else:
        agent = Agent(
            nepochs=50,
            batch_size=128,
            hidden_layers=[100, 100, 100],
            n_layers=10,
        )
        agent.run_main()