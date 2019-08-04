import hw4.inception_score as inception
from hw4.model_gan import WGANModel
from hw4.data_loader import get_data
from hw4 import save_images

import torch
import torch.autograd as autograd
import torch.optim as optim
import torch.distributions as dist
import time
import numpy as np
import argparse
import os

from utils.logger import TorchLogger

class HW:

    def __init__(self,
                 batch_size,
                 nz,
                 n_iter,
                 save_dir,
                 *,
                 load=False,
                 lr=2e-4,
                 beta1=0,
                 beta2=0.9,
                 sch_iter_rate=10000,
                 gamma=10,
                 ncritic=5,
                 log_rate=100,
                 ckpt_rate=1000,
                 ):

        self.batch_size = batch_size
        self.niter = n_iter
        self.gamma = gamma
        self.ncritic = ncritic
        self.log_rate = log_rate
        self.ckpt_rate = ckpt_rate
        self.device = torch.device('cuda:3') if torch.cuda.is_available() else 'cpu'

        if not load:
            meta_data = dict(
                batch_size=batch_size,
                n_iter=n_iter,
                ncritic=ncritic,
                nz=nz,
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                gamma=gamma
            )
            self.logger = TorchLogger(save_dir, meta_data=meta_data)

        self.model = WGANModel(nz)
        self.model.to(self.device)
        self.opt_gen = optim.Adam(self.model.gen.parameters(), lr=lr, betas=(beta1, beta2))
        self.opt_critic = optim.Adam(self.model.disc.parameters(), lr=lr, betas=(beta1, beta2))
        # ??
        self.sch_gen = optim.lr_scheduler.StepLR(self.opt_gen, sch_iter_rate, 0.3)
        self.sch_critic = optim.lr_scheduler.StepLR(self.opt_critic, sch_iter_rate, 0.3)

        # real data generator
        self.train_loader, self.test_loader, _ = get_data('./data', batch_size)
        self.gen_train = self.inf_train_gen()
        # prior on z ~ p(z)
        self.prior = dist.MultivariateNormal(torch.zeros(nz), torch.eye(nz))
        # inception score
        self.ins_score = None

        self.log(f'{"iter":<10} | {"gen_loss":<15} | {"critic_loss":<15} | '
                 f'{"time":<10}')

    @property
    def log(self):
        return self.logger.log

    def inf_train_gen(self):
        while True:
            for images, target in self.train_loader:
                yield images

    def get_inception_score(self, sample_batch=100, split=10):
        all_samples = []
        for i in range(split):
            samples_100 = torch.randn(sample_batch, 128).to(self.device)
            all_samples.append(self.model.generate(samples_100).cpu().data.numpy())

        all_samples = np.concatenate(all_samples, axis=0)
        all_samples = np.multiply(np.add(np.multiply(all_samples, 0.5), 0.5), 255).astype('int32')
        all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0, 2, 3, 1)
        return inception.get_inception_score(list(all_samples))

        # For generating samples
    def generate_image(self, frame):
        os.makedirs('./tmp/cifar10/', exist_ok=True)

        fixed_noise_128 = torch.randn(128, 128).to(self.device)
        samples = self.model.gen(fixed_noise_128)
        samples = samples.view(-1, 3, 32, 32)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.cpu().data.numpy()
        save_images.save_images(samples, f'./tmp/cifar10/samples_{frame}.jpg')

    def train(self):
        s = time.time()
        for i in range(self.niter):

            # lowering computation
            for p in self.model.gen.parameters():
                p.requires_grad = False
            for p in self.model.disc.parameters():
                p.requires_grad = True

            for critic_iter in range(self.ncritic):
                x_real = next(self.gen_train)
                nsamples = x_real.shape[0]
                x_real = x_real.to(self.device)
                z = self.prior.sample((nsamples,)).to(self.device)
                eps = np.random.rand()
                x_fake = self.model.generate(z)
                x_hat: torch.Tensor = eps * x_real + (1 - eps) * x_fake
                dw_real = self.model.discriminate(x_real)
                dw_fake = self.model.discriminate(x_fake)

                x_hat.requires_grad_(True)
                dw_hat: torch.Tensor = self.model.discriminate(x_hat)
                gp = autograd.grad(dw_hat, x_hat, torch.ones(dw_hat.size()).to(self.device),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
                gp = gp.view((self.batch_size,  -1))
                l2_dw_hat_grad = gp.norm(2, dim=-1)
                loss_critic = - dw_real.mean() + dw_fake.mean() - \
                              self.gamma * ((l2_dw_hat_grad - 1) ** 2).mean()

                self.opt_critic.zero_grad()
                loss_critic.backward()
                self.opt_critic.step()

            # lowering computation
            for p in self.model.gen.parameters():
                p.requires_grad = True
            for p in self.model.disc.parameters():
                p.requires_grad = False

            z = self.prior.sample((self.batch_size, )).to(self.device)
            x_fake = self.model.generate(z)
            dw_fake = self.model.discriminate(x_fake)
            loss_gen = - dw_fake.mean()
            self.opt_gen.zero_grad()
            loss_gen.backward()
            self.opt_gen.step()

            if i == 0 or (i + 1) % self.log_rate == 0:
                self.log(f'{i:<10} | {loss_gen.item():<15.4} | {loss_critic.item():<15.4} | '
                         f'{time.time() - s:<10.4}', show=False)
                print(f'{i} | {loss_gen.item()} | {loss_critic.item()} | {time.time() - s}')
                s = time.time()
                self.generate_image(i)

            if i == 0 or (i + 1) % self.ckpt_rate == 0:
                self.logger.save_model(self.model)

        ins_score = self.get_inception_score()
        self.log(f'{i:<10} | {loss_gen.item():<15.4} | {loss_critic.item():<15.4} | '
                 f'{ins_score[0]:<5.3}{ins_score[1]:<5.3}| '
                 f'{time.time() - s:<10.4}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir', help='directory to save the model and logs or the checkpoint in '
                                    'case of just generating samples')
    parser.add_argument('-b', '--batch', help='batch size', type=int, default=128)
    parser.add_argument('-nz', '--nz', type=int, default=128)
    parser.add_argument('-n', '--niter', help='number of iterations', type=int, default=100000)
    parser.add_argument('-l', '--load', action='store_true')
    args = parser.parse_args()

    sol = HW(batch_size=args.batch,
             nz=args.nz,
             n_iter=args.niter,
             save_dir=args.dir)
    sol.train()

