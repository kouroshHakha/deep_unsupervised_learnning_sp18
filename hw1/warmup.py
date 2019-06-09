import numpy as np
import tensorflow as tf
import pdb


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


    def train(self, train_data, niter=1000, batch_size=64):

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            for itr in range(niter):
                xdata = np.random.choice(train_data, batch_size, replace=False)
                feed_dict = {self.data_ph: xdata}
                loss_m = sess.run(self.loss, feed_dict)
                sess.run(self.update_op, feed_dict)
                loss_p = sess.run(self.loss, feed_dict)
                print(f'[info] loss- = {loss_m}, loss+ = {loss_p}')

    def main(self):
        data = self.sample_data()

        train_idx = int(len(data) * 0.8)
        train_data = data[:train_idx]
        valid_data = data[train_idx:]

        self.make_flow()
        self.train(train_data, niter=10000, batch_size=512)

if __name__ == '__main__':
    agent = WarmUp()
    agent.main()