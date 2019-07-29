import time
import argparse
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

from utils.logger import TorchLogger
from utils.data import divide_ds

import hw3.dataset as dt
from hw3.vae import VAE2D
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

class Q1:

    def __init__(self, save_directory, **kwargs):
        data_id = kwargs.get('data_id', 1)
        if data_id == 1:
            self.data = dt.sample_data_1()
        else:
            self.data = dt.sample_data_2()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

        if not kwargs.get('load', False):
            self.logger = TorchLogger(save_directory, meta_data=kwargs)

        hidden_layers = kwargs.get('hidden_layers', [20, 20, 20])
        part = kwargs.get('part', 'A')
        self.batch_size = kwargs.get('batch_size', 32)
        self.nepochs = kwargs.get('nepochs', 10)

        self.model: nn.Module = VAE2D(2, 2, hidden_layers, question=part)
        self.model.to(self.device)

        self.opt = optim.Adam(self.model.parameters())

    @property
    def log(self):
        return self.logger.log

    def run_epoch(self, data, mode='train'):

        s = time.time()
        nsamples = data.shape[0]
        b = self.batch_size
        nsteps = 1 if mode == 'test' else nsamples // b
        epoch_loss, epoch_rec_loss, epoch_reg_loss = 0, 0, 0
        for step in range(nsteps):
            xin = data[step * b: (step + 1) * b]
            xin = torch.from_numpy(xin).float()
            xin = xin.to(self.device)

            if mode == 'test':
                with torch.no_grad():
                    loss, rec_loss, reg_loss,  z, xhat = self.model(xin)
            else:
                loss, rec_loss, reg_loss, z, xhat = self.model(xin)

            epoch_loss += loss.item() / nsteps
            epoch_rec_loss += rec_loss.item() / nsteps
            epoch_reg_loss += reg_loss.item() / nsteps

            if mode == 'train':
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                if step % 500 == 0 :
                    self.log(f'step {step} -> training loss = {loss.item()}', show=False)

        if mode == 'train':
            self.logger.save_model(self.model)
        self.log(f'{mode} epoch finished in {time.time() - s} secs')
        self.log(f'loss: {epoch_loss}')
        self.log(f'rec_loss: {epoch_rec_loss}')
        self.log(f'reg_loss: {epoch_reg_loss}')

    def train(self):
        xtrain, xtest, _, _ = divide_ds(self.data, train_per=0.8)
        for epoch in range(self.nepochs):
            self.run_epoch(xtrain, mode='train')
            self.run_epoch(xtest, mode='test')
        return self.logger.path

    def show_data(self):
        dt.display_2d_hitmap(self.data, cmap='jet')

    def load(self, path: Path):
        self.model.load_state_dict(torch.load(path))

    def generate(self, n=1, decoder_random=True, variance=False):
        self.model.eval()
        with torch.no_grad():
            z_samples = self.model.prior.sample((n,))
            xhat_dist = self.model.decoder(z_samples)
            if variance:
                return xhat_dist.variance

            if decoder_random:
                samples = xhat_dist.sample()
            else:
                samples = xhat_dist.mean
        return samples

    def display_posterior(self, x, n=10000):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        if x.dim() == 1:
            x = x[None]
        self.model.eval()
        z_dist = self.model.encoder(x)
        z_samples = z_dist.sample((n,)).to('cpu').numpy()
        dt.display_2d_hitmap(z_samples.squeeze())

class Q2(Q1):

    def __init__(self, *args, **kwargs):
        Q1.__init__(self, *args, **kwargs)
        self.data, self.labels = dt.sample_data_3()

    def display_latent(self, n):
        z_dist = self.model.encoder(torch.from_numpy(self.data).to(self.device).float())
        z = z_dist.sample().to('cpu').numpy().squeeze()
        z1 = z[self.labels == 0, :]
        z2 = z[self.labels == 1, :]
        z3 = z[self.labels == 2, :]
        dt.display_2d_scatter(z1, color='r', label='0', alpha=1)
        dt.display_2d_scatter(z2, color='g', label='1', alpha=0.01)
        dt.display_2d_scatter(z3, color='b', label='2', alpha=0.01)
        plt.legend()

    def get_iwae_loss(self, x, m=1):
        self.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            loss, rec_loss, reg_loss, _, _, xprob, kl = self.model(x, analytic_kl = False)
            ana_loss, ana_rec_loss, ana_reg_loss, _, _, _, _ = self.model(x, analytic_kl = True)
            latent_dist = self.model.encoder(x)
            mu, var = latent_dist.mean, latent_dist.variance
            eps_dist = dist.MultivariateNormal(torch.zeros(2), torch.eye(2))
            torch.manual_seed(70)
            eps = eps_dist.sample((m, x.shape[0], ))
            z_samples = var * eps + mu
            # torch.manual_seed(70)
            # z_samples = latent_dist.sample((m, ))
            z_samp_reshaped = z_samples.view((-1, 2))
            x_samp_dist = self.model.decoder(z_samp_reshaped)
            vae_prob = x_samp_dist.log_prob(x.repeat(m, 1)).view(m, x.shape[0])
            posterior = latent_dist.log_prob(z_samples)
            prior = self.model.prior.log_prob(z_samples)

            inputs = vae_prob + prior - posterior
            # log_sum_exp trick
            inputs_mean = inputs.mean()
            tot = (vae_prob + prior - posterior).exp()
            # tot = -((inputs - inputs_mean).exp().mean(dim=0).log() + inputs_mean)
            x_loss = tot.mean(dim=0)
            iwae_loss = (-x_loss.log()).mean(dim=-1)
            # iwae_loss = tot.mean(dim=-1)
            pdb.set_trace()
            print(f'm = {m}, normal_loss = {loss}')
            print(f'm = {m}, ana_loss = {ana_loss}')
            print(f'm = {m}, iwae_loss = {iwae_loss}')

        return iwae_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='directory to save the model and logs or the checkpoint in '
                                    'case of just generating samples')
    parser.add_argument('q', help='question id; either A or B or 2')
    parser.add_argument('-b', '--batch', help='batch size', type=int, default=128)
    parser.add_argument('-n', '--nepoch', help='number of epochs', type=int, default=10)
    parser.add_argument('-l', '--load', action='store_true')
    parser.add_argument('-g', '--generate', action='store_true')
    parser.add_argument('-d', '--data',help='data_id, 1 for samples1 o.w. samples2', type=int,
                        default=1)
    args = parser.parse_args()

    Q_ = Q2 if args.q == '2' else Q1
    sol = Q_(args.dir,
             part=args.q,
             load=args.load,
             data_id=3 if args.q == '2' else args.q,
             batch_size=args.batch,
             nepochs=args.nepoch,
             hidden_layers=[200, 200, 200])

    if args.load:
        ckpt_path = Path(args.dir)
        if not ckpt_path.match('*.pt'):
            raise ValueError(f'path {ckpt_path} is not valid')
        sol.load(ckpt_path)
        log_path = ckpt_path.parent
        plt.clf()
        plt.subplot(111)
        sol.show_data()
        plt.savefig(log_path / 'data.png')
    else:
        sol.show_data()
        plt.show()
        log_path = sol.train()

    if args.generate:
        _, test, _, _ = divide_ds(sol.data, train_per=0.8)

        samples1 = sol.generate(n=100000, decoder_random=False).to('cpu').numpy()
        samples2 = sol.generate(n=100000, decoder_random=True).to('cpu').numpy()
        variance = sol.generate(n=100000, variance=True).to('cpu').numpy()
        plt.clf()
        plt.subplot(121)
        dt.display_2d_hitmap(samples1, cmap='jet')
        plt.title('mean of p(x|z)')
        plt.subplot(122)
        dt.display_2d_hitmap(variance, cmap='jet')
        plt.title('variance of p(x|z)')
        plt.savefig(log_path / 'mean_var.png')
        plt.subplot(111)
        dt.display_2d_hitmap(samples2, cmap='jet')
        plt.title('x~p(x|z)')
        plt.savefig(log_path / 'samples.png')
        if args.q == '2':
            plt.clf()
            plt.subplot(111)
            sol.display_latent(n=100000)
            plt.savefig(log_path / 'latent.png')
            sol.get_iwae_loss(test, m=1)
            sol.get_iwae_loss(test, m=5)
            sol.get_iwae_loss(test, m=10)
