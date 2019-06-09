import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import pdb
import matplotlib.pyplot as plt

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class WarmUp:

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

        # tensors defined later
        self.data_ph, self.p_of_x = None, None
        self.theta_params, self.probs = None, None
        self.loss, self.update_op, self.optimizer = None, None, None

    @staticmethod
    def sample_data():
        count = 10000
        a = 0.3 + 0.1 * np.random.randn(count)
        b = 0.8 + 0.05 * np.random.randn(count)
        mask = np.random.rand(count) < 0.5
        foo: np.ndarray = a * mask + b * (1 - mask)
        data = np.clip(foo, 0.0, 1.0)
        return np.digitize(data, np.linspace(0, 1, 100))

    def make_flow(self):
        tf.reset_default_graph()
        self.data_ph = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.theta_params = tf.get_variable("theta", (100,), dtype=tf.float32,
                                            initializer=tf.zeros_initializer)
        self.probs = tf.nn.softmax(self.theta_params)
        self.p_of_x = tf.gather(self.probs, self.data_ph)

        self.loss = -tf.reduce_mean(tf.log(self.p_of_x))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_op = self.optimizer.minimize(self.loss)

        # sampling operation
        dist = tfp.distributions.Categorical(probs=self.probs)
        self.get_sample_op = dist.sample()

    def train(self, sess, train_data, valid_data=None, niter=1000, batch_size=64):

        train_xvar, valid_xvar = [], []
        train_yvar, valid_yvar = [], []

        for itr in range(niter):
            xdata = np.random.choice(train_data, batch_size, replace=False)
            feed_dict = {self.data_ph: xdata}
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


    def get_pdist(self, sess):
        x = np.arange(1, 101, 1)
        p_dist = sess.run(self.probs, feed_dict={self.data_ph: x})

        samples = []
        for _ in range(1000):
            samples.append(sess.run(self.get_sample_op))

        return dict(
            x=x,
            pmodel=p_dist,
            samples=samples
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
        plt.bar(results['prob']['x'], results['prob']['pmodel'], color='r', label='model prob')
        plt.hist(results['prob']['samples'], bins=results['prob']['x'], density=True,
                 label='estimated prob', color='b', alpha=0.2)
        plt.legend()
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
            results = self.train(sess, train_data, valid_data, niter=100, batch_size=2048)
            prob_dist = self.get_pdist(sess)
            results['prob'] = prob_dist

        self.print_results(results)

if __name__ == '__main__':
    agent = WarmUp()
    agent.main()