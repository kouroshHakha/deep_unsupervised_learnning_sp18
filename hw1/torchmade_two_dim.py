from typing import Any
import torch
import torch.optim as optim
import torch.functional as F
from torch.distributions import Categorical
import torch.nn as nn
from path import Path
import numpy as np
import matplotlib.pyplot as plt
from pytorch_made.made import MADE
import os
import time
import pdb

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def to_compute_resource(tensor: torch.Tensor) -> Any:
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor

class Q2:
    def __init__(self,
                 hidden_list,
                 *,
                 num_masks=1,
                 batch_size=64,
                 learning_rate=0.001,
                 natural_ordering=True,
                 resample_every=10,
                 navg_samples=5,
                 nepochs=10
                 ):

        self.dim = 200
        self.nin = 2
        self.num_masks = num_masks
        self.hidden_list = hidden_list
        self.batch_size = batch_size
        self.lr = learning_rate
        self.natural_ordering = natural_ordering
        self.resample_every = resample_every
        self.navg_samples = navg_samples
        self.nepochs = nepochs


        self.model = None
        self.xtr, self.xte = None, None
        self.opt, self.scheduler = None, None


    def sample_data(self, nsample=100000):

        dim = self.dim
        fpath = Path(__file__)
        dist = np.load(fpath.parent / "distribution.npy")

        samples_flattened = np.random.choice(np.arange(dim*dim), nsample, replace=True,
                                             p=dist.flatten())

        samples_x1, samples_x2 = samples_flattened // dim, samples_flattened % dim
        samples = np.stack([samples_x1, samples_x2], axis=0)
        samples = samples.T
        return samples

    def run_epoch(self, split, *, navg_samples=1, resample_every=20):
        # torch.set_grad_enabled(split=='train') # enable/disable grad for efficiency of forwarding test batches
        self.model.train() if split == 'train' else self.model.eval()
        navg_samples = 1 if split == 'train' else navg_samples
        x = self.xtr if split == 'train' else self.xte
        n, d = x.shape
        nsteps = n // self.batch_size
        b = self.batch_size
        neg_likliehood = np.inf
        for step in range(nsteps):

            # fetch the next batch of data
            xb = x[step * b: step * b + b]
            # get the logits, potentially run the same batch a number of times, resampling each time
            xbhat = to_compute_resource(torch.zeros([b, self.dim * d]))

            for s in range(navg_samples):
                # perform order/connectivity-agnostic training by resampling the masks
                if step % resample_every == 0 or split in ['test']:
                    # if in test, cycle masks every time
                    self.model.update_masks()
                # forward the model
                xbhat += self.model(xb)
            xbhat /= navg_samples

            all_indices = np.arange(xbhat.shape[-1])
            mode_indices = all_indices % xb.shape[-1]

            log_prob = to_compute_resource(torch.zeros((xb.shape[0], )))
            for k in range(xb.shape[-1]):
                indices = np.where(mode_indices == k)[0]
                x_k = xbhat[:, indices]
                x_k_probs = x_k.softmax(dim=-1)
                x_k_p = x_k_probs[np.arange(b), xb[:, k].long()]
                log_prob += x_k_p.log2()

            neg_likliehood = -log_prob.mean(dim=-1)

            # backward/update
            if split == 'train':
                self.opt.zero_grad()
                neg_likliehood.backward()
                self.opt.step()

        print("%s epoch average loss: %f" % (split, neg_likliehood.item()))

    def gen_samples(self, nsamples=100000):
        self.model.eval()
        x = to_compute_resource(torch.zeros([nsamples, self.nin]))

        nout = self.nin * self.dim
        all_indices = np.arange(nout)
        mode_indices = all_indices % self.nin
        self.model.update_masks(force_natural_ordering=True)
        for k in range(x.shape[-1]):
            indices = np.where(mode_indices == k)[0]
            logits = self.model(x)[:, indices]
            xk_dist = Categorical(logits=logits)
            xk_samples = xk_dist.sample()
            x[:, k] = xk_samples[:]
        samples = x.numpy()
        return samples

    @staticmethod
    def print_results(results):
        # print log loss of both training and validation datasets
        # plt.figure(1)
        # plt.plot(results['train_x'], results['train_y'], color='r', label='train')
        # plt.plot(results['valid_x'], results['valid_y'], color='b', label='valid')
        # plt.title('negative log likelihood')
        # plt.legend()

        plt.figure(2)
        plt.hist2d(results['data'][:, 0], results['data'][:, 1], bins=200,
                   range=[[0,199], [0, 199]], cmap=plt.get_cmap('hot'))
        plt.figure(3)
        plt.hist2d(results['model'][:, 0], results['model'][:, 1], bins=200,
                   range=[[0,199], [0, 199]], cmap=plt.get_cmap('hot'))
        plt.show()

    def main(self, seed=10):

        # reproducibility is good
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print("loading data ... ")
        data = self.sample_data()
        train_idx = int(len(data) * 0.8)
        xtr = data[:train_idx]
        xte = data[train_idx:]

        self.model: nn.Module = MADE(xtr.shape[1], self.hidden_list, xtr.shape[1] * self.dim,
                                     num_masks=self.num_masks,
                                     natural_ordering=self.natural_ordering,
                                     seed=seed)
        print("number of model parameters:",
              sum([np.prod(p.size()) for p in self.model.parameters()]))

        if torch.cuda.is_available():
            self.model.cuda()
            self.xtr = torch.from_numpy(xtr).float().cuda()
            self.xte = torch.from_numpy(xte).float().cuda()
        else:
            self.xtr = torch.from_numpy(xtr).float()
            self.xte = torch.from_numpy(xte).float()

            # set up the optimizer
        self.opt = torch.optim.Adam(self.model.parameters(), self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=45, gamma=0.1)

        # start the training
        for epoch in range(self.nepochs):
            s = time.time()
            print("epoch %d" % (epoch, ))
            # self.scheduler.step(epoch)
            self.run_epoch('train', resample_every=self.resample_every)
            self.run_epoch('test')
            print(f'training time for epoch {epoch}: {time.time() - s}')

        print("optimization done. full test set eval:")
        samples = self.gen_samples(nsamples=100000)
        results = dict(
            data=data,
            model=samples,
        )
        self.print_results(results)


if __name__ == '__main__':

    q = Q2(
        [30, 30, 30, 30, 30],
        num_masks=1,
        batch_size=128,
        learning_rate=1e-3,
        natural_ordering=True,
        resample_every=1,
        navg_samples=1,
        nepochs=20,
    )
    q.main()