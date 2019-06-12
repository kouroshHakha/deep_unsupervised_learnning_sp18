from typing import List, Dict, Any, Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from path import Path
import matplotlib.pyplot as plt
import pdb
import os

class MadeTwoDim:

    def __init__(self,
                 learning_rate=0.001,
                 batch_size=64,
                 niter=1000,
                 hidden_layers=(20,20,20),
                 ):

        self.dim = 200
        self.ninputs=2
        self.lr = learning_rate
        self.batch_size = batch_size
        self.niter = niter
        self.hidden_layers = list(hidden_layers)

        # graph
        self.graph = None

        np.random.seed(0)
        tf.random.set_random_seed(0)

        # graph tensors
        self.data_ph, self.p_of_x1, self.p_of_x2_given_x1, self.p_of_x1_x2 = None, None, None, None
        self.probs_x1, self.probs_x2_given_x1 = None, None
        self.theta_param1 = None
        self.loss, self.update_op, self.optimizer = None, None, None

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


    def _get_conditional_one_matrix(self, m_minus: np.ndarray, m_plus: np.ndarray) -> np.ndarray:
        mask = np.zeros([len(m_plus), len(m_minus)])
        for i in range(len(m_plus)):
            for j in range(len(m_minus)):
                if m_plus[i] >= m_minus[j]:
                    mask[i,j] = 1
        return mask

    def _get_last_mask(self, m_minus: np.ndarray, m_plus: np.ndarray) -> np.ndarray:
        mask = np.zeros([len(m_plus), len(m_minus)])
        for i in range(len(m_plus)):
            for j in range(len(m_minus)):
                if m_plus[i] > m_minus[j]:
                    mask[i,j] = 1
        return mask

    def _get_random_masks(self, order: np.ndarray) -> List[np.ndarray]:
        """
        generates a list of all masks, randomly
        :param seed_mask:
        :return:
        """
        prev_min = 0
        mfunc_list = []
        for size_h in self.hidden_layers:
            mfunc = np.random.choice(np.arange(prev_min, self.ninputs-1), size_h)
            prev_min = np.min(mfunc)
            mfunc_list.append(mfunc)

        import pdb

        m_minus_list = [order] + mfunc_list[:-1]
        m_plus_list = mfunc_list
        mask_list = []
        for m_minus, m_plus in zip(m_minus_list, m_plus_list):
            mask_list.append(self._get_conditional_one_matrix(m_minus, m_plus))

        mask_list.append(self._get_last_mask(mfunc_list[-1], order))

        return mask_list

    def _get_random_order(self):
        indices = np.arange(self.ninputs)
        np.random.shuffle(indices)
        return indices

    def train(self,
              sess,
              train_data,
              valid_data=None,
              niter=1000,
              batch_size=64,
              randomize_order=True,
              randomize_mask=True,
              ) -> Dict[str, Any]:

        train_xvar, valid_xvar = [], []
        train_yvar, valid_yvar = [], []

        indices = np.arange(len(train_data))

        order = self._get_random_order()
        masks = self._get_random_masks(order)
        for itr in range(niter):
            xdata_idx = np.random.choice(indices, batch_size, replace=False)
            xdata = train_data[xdata_idx, :]

            feed_dict={
                self.data_ph: xdata,
                self.order: order,
            }
            feed_dict.update(dict(zip(self.masks, masks)))
            _, train_loss = sess.run([self.update_op, self.loss], feed_dict)

            if randomize_order:
                order = self._get_random_order()
            if randomize_mask:
                masks = self._get_random_masks(order)


    def _make_masked_p(self,
                         input: tf.Tensor,
                         mask: tf.Tensor,
                         units: int,
                         activation: Callable = None,
                         weight_initilizer: Callable = tf.random_normal_initializer,
                         bias_initilizer: Callable = tf.zeros_initializer,
                         name: str = None) -> tf.Tensor:

        with tf.name_scope(name):
            weights = tf.get_variable(shape=(units, input.shape[-1]), dtype=tf.float32,
                                      initializer=weight_initilizer, name='weight')
            bias = tf.get_variable(shape=(units, ), dtype=tf.float32,
                                   initializer=bias_initilizer, name='bias')
            output = activation(tf.matmul(weights * mask, input) + bias)

        return output

    def _make_masked_mlp(self,
                         input: tf.Tensor,
                         masks: List[tf.Tensor],
                         hidden_layers: List[int],
                         mid_activation: Callable = tf.nn.relu,
                         name=''):

        assert len(hidden_layers) + 1 == len(masks), 'number of masks doesn\'t match the number ' \
                                                     'of hidden layers'
        layer = input
        for i, (mask, size_h) in enumerate(zip(masks[:-1], hidden_layers)):
            layer = self._make_masked_p(layer, mask, size_h, mid_activation, name=f'layer_{i}')

        # TODO: this is built for binary output, doesn't support non binary
        output = self._make_masked_p(layer, masks[-1], self.ninputs, activation=tf.nn.softmax,
                                     name=f'layer_out}')

        return output

    def make_flow(self):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.data_ph = tf.placeholder(dtype=tf.int32, shape=(None, self.ninputs), name='input')

        self.masks = []
        prev_h_list = [self.ninputs] + self.hidden_layers
        next_h_list = self.hidden_layers + [self.ninputs]
        for i, (prev_h, next_h) in enumerate(zip(prev_h_list, next_h_list)):
            self.masks.append(tf.placeholder(dtype=tf.bool,
                                             shape=(next_h, prev_h),
                                             name=f'mask_{i}'))

        self.order = tf.placeholder(dtype=tf.int32, shape=(self.ninputs,), name='input_order')

        # hacky way to slice a tensor using another tensor
        one_hot_order = tf.one_hot(self.order, depth=self.ninputs)
        self.input_shuff = tf.reduce_sum(self.data_ph * one_hot_order)

        self.output = self._make_masked_mlp(self.input_shuff, self.masks,
                                            hidden_layers=self.hidden_layers,
                                            mid_activation=tf.nn.relu,
                                            name='model')

        self.loss = -tf.reduce_sum(tf.math.log(self.output), axis=-1)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_op = self.optimizer.minimize(self.loss)


    def main(self):
        data = self.sample_data()

        train_idx = int(len(data) * 0.8)
        train_data = data[:train_idx]
        valid_data = data[train_idx:]


        self.make_flow()

if __name__ == '__main__':
    agent = MadeTwoDim(hidden_layers=(3,3))
    agent.main()