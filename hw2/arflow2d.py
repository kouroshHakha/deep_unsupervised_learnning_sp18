import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np
from utils.data import divide_ds
from hw1.pytorch_made.made import MADE
from utils.logger import TorchLogger

import pdb
import os
import sys

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = TorchLogger('data')

class Model(MADE):

    def forward(self, x):
        yhat = MADE.forward(self, x)
        x0_params = yhat[:, ::2]
        x1_params = yhat[:, 1::2]

        mu_x0 = x0_params[:, ::3]
        sigma_x0 = x0_params[:, 1::3]
        sigma_x0 = nn.ReLU()(sigma_x0) + torch.ones(sigma_x0.shape) * 0.1
        pi_x0: torch.Tensor= x0_params[:, 2::3]
        pi_x0 = pi_x0.softmax(dim=-1)

        x0_dist = Normal(mu_x0, sigma_x0)
        x0 = x[:, 0].float()
        df_x0 = pi_x0 * torch.exp(x0_dist.log_prob(x0[:, None]))
        z0 =  (pi_x0 * x0_dist.cdf(x0[:, None])).sum(dim=-1)
        jac0 = df_x0.sum(dim=-1)

        mu_x1 = x1_params[:, ::3]
        sigma_x1: torch.Tensor = x1_params[:, 1::3]
        sigma_x1 = nn.ReLU()(sigma_x1) + torch.ones(sigma_x1.shape) * 0.1
        pi_x1 = x1_params[:, 2::3]
        pi_x1 = pi_x1.softmax(dim=-1)

        x1_dist = Normal(mu_x1, sigma_x1)
        x1 = x[:, 1].float()
        df_x1 = pi_x1 * torch.exp(x1_dist.log_prob(x1[:, None]))
        z1 =  (pi_x1 * x1_dist.cdf(x1[:, None])).sum(dim=-1)
        jac1 = df_x1.sum(dim=-1)

        ll = jac0.log() + jac1.log()
        z = torch.cat([z0[:, None], z1[:, None]], dim=-1)

        return yhat, ll, z

def sample_data():
    count = 100000
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]

    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]


def run_epoch(model, optimizer, data,
              batch_size, k, mode='train'):
    model.train(mode == 'train')

    ndata, nin = data.shape
    nout = 3 * k * nin
    b = batch_size
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    nsteps = ndata // b
    for step in range(nsteps):
        xbatch = data[step: (step + 1) * b].to(device)
        _, ll, _ = model(xbatch)

        nll = -torch.mean(ll, dim=-1)
        if mode == 'train':
            optimizer.zero_grad()
            nll.backward()
            optimizer.step()

        if step % 100 == 0 and mode == 'train':
            print(f'[train] neg_ll = {nll}')

    print(f'[{mode}] epoch finished nll = {nll}')
    logger.log(f'[{mode}] epoch finished nll = {nll}')

def run_main(k=5, batch_size=32):
    data_x, data_y = sample_data()

    train_x, test_x, train_y, test_y = divide_ds(data_x, data_y, 0.8)
    train_x = torch.from_numpy(train_x)
    test_x = torch.from_numpy(test_x)

    ndata, nin = train_x.shape
    nout = 3 * k * nin
    model: nn.Module = Model(nin, [20, 20, 20], nout, natural_ordering=True, bias_init=1.0)
    optimizer = optim.Adam(model.parameters())

    model.train(True)
    for epoch in range(10):
        print(f'epoch = {epoch}')
        run_epoch(model, optimizer, train_x, batch_size, k, mode='train')
        run_epoch(model, optimizer, test_x, batch_size, k, mode='test')
        logger.save_model(model)


def show_density(k, fpath):

    nin = 2
    nout = 3 * k * nin
    model: nn.Module = Model(nin, [20, 20, 20], nout, natural_ordering=True, bias_init=1.0)
    model.load_state_dict(torch.load(fpath))

    num = 100
    x = np.linspace(-4, 4, num)
    x_array = np.transpose([np.tile(x, len(x)), np.repeat(x, len(x))])
    _, y_arr, _ = model(torch.from_numpy(x_array))
    y_arr = torch.detach(y_arr.to('cpu')).numpy()
    pdb.set_trace()

def show_latent(k, fpath):

    nin = 2
    nout = 3 * k * nin
    model: nn.Module = Model(nin, [20, 20, 20], nout, natural_ordering=True, bias_init=1.0)
    model.load_state_dict(torch.load(fpath))

    data_x, data_y = sample_data()

    train_x, test_x, train_y, test_y = divide_ds(data_x, data_y, 0.8)
    train_x = torch.from_numpy(train_x)
    test_x = torch.from_numpy(test_x)

    _, _, z_var = model(test_x)
    z_var = torch.detach(z_var.to('cpu')).numpy()

    colors = ['r', 'b', 'g']
    for i in range(3):
        color = colors[i]
        z = z_var[test_y == i, :]
        pdb.set_trace()

if __name__ == '__main__':
    run_main(k=5, batch_size=32)
    # show_density(k=5, fpath=sys.argv[1])
    # show_latent(k=5, fpath=sys.argv[1])