from typing import cast
import torch
import torch.optim as optim
import torch.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
import pdb

def get_samples(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':

    file = sys.argv[1]
    data = get_samples(file)
    dtrain = cast(np.ndarray, data['train']).transpose([0,3,1,2])
    dtest = cast(np.ndarray, data['test']).transpose([0,3,1,2])

    xtrain = torch.from_numpy(dtrain)
    xtest = torch.from_numpy(dtest)
    conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, dilation=1, groups=1, bias=True,
                      padding_mode='zeros')

    xin = xtrain[0][None, ...].float()
    xin.requires_grad_(True)
    conv1_out = conv1(xin)
    conv1_out[0, 0, 14, 14].backward()
    grad = xin.grad[0].numpy()
    foo = np.where(grad[2] != 0)
    print(foo)
    pdb.set_trace()
