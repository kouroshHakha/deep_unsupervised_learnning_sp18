import numpy as np
import matplotlib.pyplot as plt

def sample_data_1():
    count = 100000
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + rand.randn(count, 2) * [[5.0, 1.0]]
def sample_data_2():
    count = 100000
    rand = np.random.RandomState(0)
    return [[1.0, 2.0]] + (rand.randn(count, 2) * [[5.0, 1.0]]).dot(
        [[np.sqrt(2) / 2, np.sqrt(2) / 2], [-np.sqrt(2) / 2, np.sqrt(2) / 2]])
def sample_data_3():
    count = 100000
    rand = np.random.RandomState(0)
    a = [[-1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    b = [[1.5, 2.5]] + rand.randn(count // 3, 2) * 0.2
    c = np.c_[2 * np.cos(np.linspace(0, np.pi, count // 3)),
              -np.sin(np.linspace(0, np.pi, count // 3))]
    c += rand.randn(*c.shape) * 0.2
    data_x = np.concatenate([a, b, c], axis=0)
    data_y = np.array([0] * len(a) + [1] * len(b) + [2] * len(c))
    perm = rand.permutation(len(data_x))
    return data_x[perm], data_y[perm]

def display_2d_scatter(data, **kwargs):
    plt.plot(data[:, 0], data[:, 1], '.', **kwargs)

def display_2d_hitmap(data, nbins=100, **kwargs):
    z, x0_edges, x1_edges = np.histogram2d(data[:, 0], data[:, 1], bins=nbins,
                                           density=True)
    z = z.T
    fig = plt.figure(1)
    ax = fig.gca()
    ax.imshow(z, interpolation='gaussian', origin='low',
              extent=[x0_edges[0], x0_edges[-1], x1_edges[0], x1_edges[-1]], **kwargs)