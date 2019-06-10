import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from path import Path
import pdb
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TwoDimension:

    def __init__(self,
                 learning_rate=0.03,
                 batch_size=64,
                 niter=1000,
                 ):

        self.lr = learning_rate
        self.batch_size = batch_size
        self.niter = niter

        # init random seed
        np.random.seed(0)

    @staticmethod
    def sample_data(nsample=100000):

        dim = 200
        fpath = Path(__file__)
        dist = np.load(fpath.parent / "distribution.npy")

        samples_flattened = np.random.choice(np.arange(dim*dim), nsample, replace=True,
                                             p=dist.flatten())

        samples_x1, samples_x2 = samples_flattened // dim, samples_flattened % dim
        samples = np.stack([samples_x1, samples_x2], axis=0)
        samples = samples.T
        return samples

    def main(self):
        data = self.sample_data()

        train_idx = int(len(data) * 0.8)
        train_data = data[:train_idx]
        valid_data = data[train_idx:]

if __name__ == '__main__':
    agent = TwoDimension()

    data = agent.sample_data()
    print(data)
    pdb.set_trace()
