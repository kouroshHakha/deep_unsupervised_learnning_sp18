from typing import List, Dict, Any, Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from path import Path
import matplotlib.pyplot as plt
import pdb
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
        prev_min = 0
        mfunc_list = []
        for size_h in self.hidden_layers:
            mfunc = np.random.choice(np.arange(prev_min, self.ninputs-1), size_h)
            prev_min = np.min(mfunc)
            mfunc_list.append(mfunc)

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
              ):

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

            train_xvar.append(itr)
            train_yvar.append(train_loss)

            if randomize_order:
                order = self._get_random_order()
            if randomize_mask:
                masks = self._get_random_masks(order)

            if itr % (niter // 10) == 0:
                print(f'[info] iter {itr}: train_loss={train_loss}')
                if valid_data is not None:
                    feed_dict={
                        self.data_ph: xdata,
                        self.order: order,
                    }
                    feed_dict.update(dict(zip(self.masks, masks)))
                    valid_loss = sess.run(self.loss, feed_dict=feed_dict)
                    valid_xvar.append(itr)
                    valid_yvar.append(valid_loss)

        results = dict(
            train_x=train_xvar,
            train_y=train_yvar,
            valid_x=valid_xvar,
            valid_y=valid_yvar
        )
        return masks, order, results

    def _make_masked_p(self,
                       input: tf.Tensor,
                       mask: tf.Tensor,
                       units: int,
                       activation: Callable = None,
                       weight_initilizer: Callable = tf.random_normal_initializer,
                       bias_initilizer: Callable = tf.zeros_initializer,
                       name: str = None,
                       last_layer: bool =False) -> tf.Tensor:

        with tf.name_scope(name):
            weights = tf.get_variable(shape=(input.shape[-1], units), dtype=tf.float32,
                                      initializer=weight_initilizer, name=f'{name}/weight')
            bias = tf.get_variable(shape=(units, ), dtype=tf.float32,
                                   initializer=bias_initilizer, name=f'{name}/bias')
            if not last_layer:
                output = activation(tf.matmul(input, weights * tf.transpose(mask)) + bias)
            else:
                input2 = input[:, None] * mask
                input_reshaped = tf.reshape(input2, shape=[-1, input.shape[-1]])
                output = tf.matmul(input_reshaped, weights) + bias
                output_reshaped = tf.reshape(output, shape=[-1, mask.shape[0] , units])
                output = tf.nn.softmax(output_reshaped)

        return output

    def _make_masked_mlp(self,
                         input: tf.Tensor,
                         masks: List[tf.Tensor],
                         hidden_layers: List[int],
                         mid_activation: Callable = tf.nn.relu,
                         name=''):

        assert len(hidden_layers) + 1 == len(masks), 'number of masks doesn\'t match the number ' \
                                                     'of hidden layers'
        layers = []
        layer = tf.cast(input, tf.float32)
        layers.append(layer)
        for i, (mask, size_h) in enumerate(zip(masks[:-1], hidden_layers)):
            layer = self._make_masked_p(layer, mask, size_h, mid_activation, name=f'layer_{i}')
            layers.append(layer)

        output = self._make_masked_p(layer, masks[-1], self.dim, name=f'layer_out',
                                     last_layer=True)
        layers.append(output)

        return output, layers

    def make_flow(self):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.data_ph = tf.placeholder(dtype=tf.int32, shape=(None, self.ninputs), name='input')

        self.masks = []
        prev_h_list = [self.ninputs] + self.hidden_layers
        next_h_list = self.hidden_layers + [self.ninputs]
        for i, (prev_h, next_h) in enumerate(zip(prev_h_list, next_h_list)):
            self.masks.append(tf.placeholder(dtype=tf.float32,
                                             shape=(next_h, prev_h),
                                             name=f'mask_{i}'))

        self.order = tf.placeholder(dtype=tf.int32, shape=(self.ninputs,), name='input_order')

        # hacky way to slice a tensor using another tensor
        one_hot_order = tf.one_hot(self.order, depth=self.ninputs)

        self.input_shuff = tf.reduce_sum(tf.cast(self.data_ph[:, :, None], tf.float32) *
                                         one_hot_order, axis=-1)

        self.output, self.layers = self._make_masked_mlp(self.input_shuff, self.masks,
                                                         hidden_layers=self.hidden_layers,
                                                         mid_activation=tf.nn.tanh,
                                                         name='model')

        self.one_hot_mask = tf.one_hot(self.data_ph, depth=self.dim)
        self.joint_prob = tf.reduce_sum(self.output * self.one_hot_mask, axis=-1)

        self.log_likleihood = tf.math.log(self.joint_prob)
        self.neg_log_likleihood = -tf.reduce_sum(self.log_likleihood, axis=-1)
        self.loss = tf.reduce_mean(self.neg_log_likleihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_op = self.optimizer.minimize(self.loss)

        # sampling
        dist_x1 = tfp.distributions.Categorical(probs=self.output[:, 0])
        self.get_x1_sample_op = dist_x1.sample()
        dist_x2 = tfp.distributions.Categorical(probs=self.output[:, 1])
        self.get_x2_sample_op = dist_x2.sample()

    def get_samples(self, sess, order, mask, nsamples=100000):
        # xrange = np.arange(200)
        # feed_dict={
        #     self.data_ph: xr,
        #     self.order: order,
        # }
        # feed_dict.update(dict(zip(self.masks, masks)))
        # x1_dist = sess.run(self.get_x1_sample_op, feed_dict=)
        # x1_samples = np.random.choice(xrange, nsamples, replace=True, p=x1_dist)
        # dummy_data = np.stack([x1_samples, x1_samples], axis=-1)
        # x2_samples = sess.run(self.get_x2_sample_op, feed_dict={self.data_ph: dummy_data})
        # samples = np.stack([x1_samples, x2_samples], axis=-1)
        samples = None
        return samples

    @staticmethod
    def print_results(results):
        # print log loss of both training and validation datasets
        plt.figure(1)
        plt.plot(results['train_x'], results['train_y'], color='r', label='train')
        plt.plot(results['valid_x'], results['valid_y'], color='b', label='valid')
        plt.title('negative log likelihood')
        plt.legend()

        # plt.figure(2)
        # plt.hist2d(results['data'][:, 0], results['data'][:, 1], bins=200,
        #            range=[[0,199], [0, 199]], cmap=plt.get_cmap('hot'))
        # plt.figure(3)
        # plt.hist2d(results['model'][:, 0], results['model'][:, 1], bins=200,
        #            range=[[0,199], [0, 199]], cmap=plt.get_cmap('hot'))
        plt.show()

    def main(self):
        data = self.sample_data()

        train_idx = int(len(data) * 0.8)
        train_data = data[:train_idx]
        valid_data = data[train_idx:]

        self.make_flow()
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # deliverables
            fmask, forder, results = self.train(sess,
                                                train_data,
                                                valid_data,
                                                niter=500,
                                                batch_size=128,
                                                randomize_mask=False,
                                                randomize_order=False)

            results['model'] = self.get_samples(sess, forder, fmask)

        self.print_results(results)


if __name__ == '__main__':
    agent = MadeTwoDim(hidden_layers=(300,400,400))
    agent.main()