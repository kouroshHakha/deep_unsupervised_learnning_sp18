from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from path import Path
import matplotlib.pyplot as plt
import pdb
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class TwoDimensionAR:

    def __init__(self,
                 learning_rate=0.001,
                 batch_size=64,
                 niter=1000,
                 ):

        self.dim = 200
        self.lr = learning_rate
        self.batch_size = batch_size
        self.niter = niter

        # graph
        self.graph = None

        # init random seed
        np.random.seed(0)
        tf.random.set_random_seed(0)

        # graph tensors
        self.data_ph, self.p_of_x1, self.p_of_x2_given_x1, self.p_of_x1_x2 = None, None, None, None
        self.probs_x1, self.probs_x2_given_x1 = None, None
        self.theta_param1 = None
        self.loss, self.update_op, self.optimizer = None, None, None


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

    def make_x2_model(self, input_tensor: tf.Tensor, output_size: int,  hidden_size_list: List[int],
                      name: str) -> tf.Tensor:
        with tf.name_scope(name):
            input_rank = len(input_tensor.shape.as_list())
            if input_rank == 1:
                layer = tf.expand_dims(input_tensor, axis=-1)
            else:
                layer = input_tensor

            layer = tf.cast(layer, dtype=tf.float32)
            # layer = tf.one_hot(layer, depth=self.dim, axis=-1)

            for i, hsize in enumerate(hidden_size_list):
                layer = tf.layers.dense(layer, hsize, activation=tf.nn.tanh, name=f'layer_{i}')
            self.output_logits = tf.layers.dense(layer, output_size, name='out_layer')
            output_prob = tf.nn.softmax(self.output_logits)

        return output_prob


    def make_flow(self):
        tf.reset_default_graph()
        self.graph = tf.get_default_graph()
        self.data_ph = tf.placeholder(dtype=tf.int32, shape=(None, 2), name='input')
        self.theta_param1 = tf.get_variable("theta1", (self.dim,), dtype=tf.float32,
                                            initializer=tf.zeros_initializer)
        self.probs_x1 = tf.nn.softmax(self.theta_param1, name='x1')
        self.p_of_x1 = tf.gather(self.probs_x1, self.data_ph[:, 0])
        with tf.name_scope('x2'):
            self.probs_x2_given_x1 = self.make_x2_model(self.data_ph[:, 0],
                                                        output_size=self.dim,
                                                        hidden_size_list=[20, 20, 20],
                                                        name='theta2')

            self.mask = tf.one_hot(self.data_ph[:, 1], self.dim)
            self.p_of_x2_given_x1 = tf.reduce_sum(tf.multiply(self.mask, self.probs_x2_given_x1),
                                                  axis=-1)

        self.p_of_x1_x2 = tf.multiply(self.p_of_x1, self.p_of_x2_given_x1, name='x1_x2')

        self.loss = -tf.reduce_mean(tf.log(self.p_of_x1_x2), name='loss')

        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
            self.update_op = self.optimizer.minimize(self.loss)

        # sampling
        dist = tfp.distributions.Categorical(probs=self.probs_x2_given_x1)
        self.get_x2_sample_op = dist.sample()


    def train(self, sess, train_data, valid_data=None, niter=1000, batch_size=64):

        train_xvar, valid_xvar = [], []
        train_yvar, valid_yvar = [], []

        indices = np.arange(len(train_data))
        for itr in range(niter):
            xdata_idx = np.random.choice(indices, batch_size, replace=False)
            xdata = train_data[xdata_idx, :]
            feed_dict = {self.data_ph: xdata}
            # pdb.set_trace()
            _, train_loss = sess.run([self.update_op, self.loss], feed_dict)
            train_xvar.append(itr)
            train_yvar.append(train_loss)

            if itr % (niter // 10) == 0:
                print(f'[info] iter {itr}: train_loss={train_loss}')
                if valid_data is not None:
                    valid_loss = sess.run(self.loss, feed_dict={self.data_ph: valid_data})
                    valid_xvar.append(itr)
                    valid_yvar.append(valid_loss)

        return dict(
            train_x=train_xvar,
            train_y=train_yvar,
            valid_x=valid_xvar,
            valid_y=valid_yvar
        )


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

    def get_samples(self, sess, nsamples=100000):
        xrange = np.arange(200)
        x1_dist = sess.run(self.probs_x1)
        x1_samples = np.random.choice(xrange, nsamples, replace=True, p=x1_dist)
        dummy_data = np.stack([x1_samples, x1_samples], axis=-1)
        x2_samples = sess.run(self.get_x2_sample_op, feed_dict={self.data_ph: dummy_data})
        samples = np.stack([x1_samples, x2_samples], axis=-1)
        return samples

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
            results = self.train(sess, train_data, valid_data, niter=500, batch_size=256)
            results['data'] = data
            results['model'] = self.get_samples(sess)

        self.print_results(results)

if __name__ == '__main__':
    agent = TwoDimensionAR()
    agent.main()