from typing import List, Dict, Any, Callable

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from path import Path
import matplotlib.pyplot as plt
import pdb
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def log(tensor, base):
    numerator = tf.log(tensor)
    denominator = tf.log(tf.constant(base, dtype=numerator.dtype))
    return numerator / denominator

class MadeTwoDim:

    def __init__(self,
                 learning_rate=0.001,
                 batch_size=64,
                 niter=1000,
                 hidden_layers=(20,20,20),
                 valid_rate=50,
                 activation=tf.nn.relu,
                 randomize_order=False,
                 randomize_masks=False
                 ):

        self.ninputs=2
        self.dim = 200
        self.lr = learning_rate
        self.batch_size = batch_size
        self.niter = niter
        self.hidden_layers = hidden_layers
        self.valid_rate = valid_rate
        self.activation = activation
        self.randomize_order = randomize_order
        self.randomize_masks = randomize_masks

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

    def _make_masked_p(self,
                       input: tf.Tensor,
                       mask: tf.Tensor,
                       units: int,
                       activation: Callable = None,
                       weight_initilizer: Callable = None,
                       bias_initilizer: Callable = tf.zeros_initializer,
                       name: str = None,
                       last_layer: bool =False) -> tf.Tensor:

        if weight_initilizer is None:
            prev_size = input.shape.as_list()[-1]
            weight_initilizer = tf.random_normal_initializer(0, np.sqrt(2 / (prev_size + units)))
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
                         name=''):

        assert len(hidden_layers) + 1 == len(masks), 'number of masks doesn\'t match the number ' \
                                                     'of hidden layers'
        layers = []
        layer = tf.cast(input, tf.float32)
        layers.append(layer)
        for i, (mask, size_h) in enumerate(zip(masks[:-1], hidden_layers)):
            layer = self._make_masked_p(layer, mask, size_h, self.activation, name=f'layer_{i}')
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
                                                         name='model')

        self.one_hot_mask = tf.one_hot(self.data_ph, depth=self.dim)
        self.joint_prob = tf.reduce_sum(self.output * self.one_hot_mask, axis=-1)

        self.neg_log_likleihood = -tf.reduce_sum(log(self.joint_prob, 2), axis=-1)
        self.loss = tf.reduce_mean(self.neg_log_likleihood)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_op = self.optimizer.minimize(self.loss)

        # sampling
        dist = tfp.distributions.Categorical(probs=self.output)
        self.get_sample_op = dist.sample()

    def train(self,
              sess,
              train_data,
              valid_data=None,
              randomize_order=True,
              randomize_mask=True,
              *,
              niter=1000,
              batch_size=64,
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

            train_loss = sess.run(self.loss, feed_dict)

            train_xvar.append(itr)
            train_yvar.append(train_loss)

            if randomize_order:
                order = self._get_random_order()
            if randomize_mask:
                masks = self._get_random_masks(order)

            if itr % self.valid_rate == 0:
                print_state = f'[info] iter {itr}: train_loss={train_loss}'
                if valid_data is not None:
                    feed_dict={
                        self.data_ph: xdata,
                        self.order: order,
                    }
                    feed_dict.update(dict(zip(self.masks, masks)))
                    valid_loss = sess.run(self.loss, feed_dict=feed_dict)
                    valid_xvar.append(itr)
                    valid_yvar.append(valid_loss)
                    print_state += f', valid_loss={valid_loss}'

                print(print_state)
            # update after printing losses
            sess.run(self.update_op, feed_dict)

        results = dict(
            train_x=train_xvar,
            train_y=train_yvar,
            valid_x=valid_xvar,
            valid_y=valid_yvar
        )

        # make these None so sampling knows what to do
        if randomize_order:
            order = None
        if randomize_mask:
            masks = None

        return masks, order, results

    def get_samples(self, sess, order=None, masks=None, *,  nsamples=100000, nruns_per_sample=1):
        # instead of averaging the probablities of each dimension while changing mask and order
        # we generate samples with random combinations. (we don't sample according to the average
        # of probabilities, we randomly change the inputs that change the probs)
        print(f'sampling {nsamples} vectors from the model ... ')
        s = time.time()

        order_shuffling = order is None
        mask_shuffling = masks is None

        if order_shuffling:
            order_list = [self._get_random_order() for _ in range(nruns_per_sample)]
            masks_list = [self._get_random_masks(order_list[i]) for i in range(nruns_per_sample)]
        elif not order_shuffling and mask_shuffling:
            order_list = [order for _ in range(nruns_per_sample)]
            masks_list = [self._get_random_masks(order_list[i]) for i in range(nruns_per_sample)]
        else:
            order_list = [order]
            masks_list = [masks]

        nsamples_arr = np.random.choice(np.arange(nruns_per_sample), nsamples, replace=True)
        nsamples_list = [np.sum(nsamples_arr == i) for i in range(nruns_per_sample)]

        samples_list = []
        for order, masks, sample_size in zip(order_list, masks_list, nsamples_list):
            xdata = np.zeros([sample_size, self.ninputs])
            dim_samples = []
            for i in order:
                feed_dict={
                    self.data_ph: xdata,
                    self.order: order,
                }
                feed_dict.update(dict(zip(self.masks, masks)))
                samples = sess.run(self.get_sample_op, feed_dict=feed_dict)
                var_samples = samples[:, i]

                dim_samples.append(var_samples)
                xdata[:, i] = var_samples

            re_order = np.argsort(order)
            dim_samples = np.array(dim_samples)
            samples_list.append(np.transpose(dim_samples[re_order]))

        samples = np.concatenate(samples_list)
        print(f'finished sampling in {time.time() - s} seconds.')
        return samples

    @staticmethod
    def print_results(results):
        # print log loss of both training and validation datasets
        plt.figure(1)
        plt.plot(results['train_x'], results['train_y'], color='r', label='train')
        plt.plot(results['valid_x'], results['valid_y'], color='b', label='valid')
        plt.title('negative log likelihood')
        plt.legend()

        plt.figure(2)
        plt.hist2d(results['data'][:, 0], results['data'][:, 1], bins=200,
                   range=[[0,199], [0, 199]], cmap=plt.get_cmap('hot'))
        plt.figure(3)
        plt.hist2d(results['model'][:, 0], results['model'][:, 1], bins=200,
                   range=[[0,199], [0, 199]], cmap=plt.get_cmap('hot'))
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
            fmasks, forder, results = self.train(sess,
                                                 train_data,
                                                 valid_data,
                                                 niter=self.niter,
                                                 batch_size=self.batch_size,
                                                 randomize_mask=self.randomize_masks,
                                                 randomize_order=self.randomize_order)
            results['data'] = data
            results['model'] = self.get_samples(sess, forder, fmasks,
                                                nsamples=100000,
                                                nruns_per_sample=5)

        self.print_results(results)


if __name__ == '__main__':
    agent = MadeTwoDim(
        learning_rate=1e-4,
        niter=5000,
        batch_size=256,
        valid_rate = 50,
        activation=tf.nn.relu,
        hidden_layers=[200, 200, 200, 200, 200],
        randomize_order=False,
        randomize_masks=False,
    )
    agent.main()