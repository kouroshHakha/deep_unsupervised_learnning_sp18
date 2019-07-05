
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
import numpy as np

from utils.data import divide_ds
from utils.logger import TorchLogger

from realnvp2d import realNVP
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
    count = 100000
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
                 ):

        self.batch_size = batch_size
        self.hidden_layers = list(hidden_layers)
        self.n_layers = n_layers
        self.nepochs = nepochs

        self.optimizer = None
        self.model = None

        meta_data = dict(
            batch_size=batch_size,
            hidden_layers=hidden_layers,
            n_layers=n_layers,
        )
        self.logger = TorchLogger('data', meta_data)

    def run_epoch(self, data, device, mode='train'):
        self.model.train(mode == 'train')
        ndata, nin = data.shape
        b = self.batch_size
        nsteps = ndata // b
        for step in range(nsteps):
            xbatch = data[step * b:(step + 1) * b].to(device)
            z, ll = self.model(xbatch)
            nll = -torch.mean(ll, dim=-1)
            if mode == 'train':
                self.optimizer.zero_grad()
                nll.backward()
                # foo = list(self.model.children())[0]
                # foo = list(foo.children())[0]
                # foo = list(foo.children())[0]
                # foo = list(foo.children())[0]
                # weight = list(foo.children())[0].weight
                # if torch.isnan(weight.grad[0,0]):
                #     pdb.set_trace()
                #     self.model(xbatch, stop=True)
                #     pdb.set_trace()
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


if __name__ == '__main__':

    agent = Agent(
        nepochs=50,
        batch_size=128,
        hidden_layers=[100, 100, 100],
        n_layers=10,
    )
    agent.run_main()