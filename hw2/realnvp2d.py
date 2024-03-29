import torch
import torch.nn as nn
import torch.nn.utils as utils

import numpy as np
import pdb

class SequentialExt(nn.Sequential):

    def forward(self, input, *args, **kwargs):
        ll = None
        for module in self._modules.values():
            input, ll = module(input, *args, sldj=ll, **kwargs)
        return input, ll


class STModel(nn.Module):

    def __init__(self, nin, nout, hidden_layers):
        nn.Module.__init__(self)

        layers = []
        sizes = [nin] + hidden_layers + [nout]

        for xminus1, x in zip(sizes[:-2], sizes[1:-1]):
            layers.append(utils.weight_norm(nn.Linear(xminus1, x, bias=True)))
            layers += [nn.BatchNorm1d(x), nn.ReLU()]

        layers.append(nn.Linear(hidden_layers[-1], nout, bias=True))
        self.net: nn.Module = nn.Sequential(*layers)

    def forward(self, x, stop=False):
        y = self.net(x)
        return y

class CouplingLayer1D(nn.Module):

    def __init__(self, mid_hidden_layers, mask_reversed=False):
        nn.Module.__init__(self)

        self.st_model: nn.Module = STModel(1, 2, mid_hidden_layers)
        self.mask_reversed = mask_reversed


    def forward(self, x, sldj=None, reverse=False, stop=False):
        if sldj is None:
            sldj = torch.zeros((x.shape[0],), device=x.device)

        if isinstance(sldj, str):
            pdb.set_trace()

        x0 = x[:, 0][:, None]
        x1 = x[:, 1][:, None]
        if not self.mask_reversed:
            st = self.st_model(x0, stop)
            s = torch.tanh(st[:, 0])[:, None]
            t = st[:, 1][:, None]
            if not reverse:
                y = torch.exp(s) * x1 + t
            else:
                y = (x1 - t) * torch.exp(-s)
            output = torch.cat([x0, y], dim=-1)

        else:
            st = self.st_model(x1, stop)
            s = torch.tanh(st[:, 0])[:, None]
            t = st[:, 1][:, None]

            if not reverse:
                y = torch.exp(s) * x0 + t
            else:
                y = (x0 - t) * torch.exp(-s)
            output = torch.cat([y, x1], dim=-1)

        sldj += s.sum(-1)
        if torch.isnan(sldj).any():
            raise RuntimeError('Sum of log of determinant of jacobian has NaN entries')
        return output, sldj

class realNVP(nn.Module):

    def __init__(self, hidden_layers, n_layers, eps=1e-4):
        nn.Module.__init__(self)

        layers = []
        for i in range(n_layers):
            layers.append(CouplingLayer1D(hidden_layers, mask_reversed=(i % 2 == 0)))

        self.net: nn.Module = SequentialExt(*layers)
        self.eps = eps

    def forward(self, x, reverse=False, stop=False):
        if not reverse:
            y, sldj = self.net(x, stop=stop)
            z = torch.sigmoid(y)
            if stop:
                pdb.set_trace()
            dlast_layer = (torch.log(z * (1 - z) + self.eps)).sum(-1)
            sldj += dlast_layer
            if torch.isnan(sldj).any():
                raise RuntimeError('Sum of log of determinant of jacobian has NaN entries')
            return z, sldj
        else:
            y = torch.log(x / (1 - x))
            z, _ = self.net(y, reverse=reverse, stop=stop)
            return z

if __name__ == '__main__':

    model: nn.Module = realNVP([200, 200, 100], 10)
    model.eval()

    x = np.array([[1, 1], [0.8, 0.8]])
    x = torch.from_numpy(x).float()
    y, sldj = model(x)
    z = model(y, reverse=True)

    pdb.set_trace()
