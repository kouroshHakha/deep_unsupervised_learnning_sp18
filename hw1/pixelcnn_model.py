import torch.nn as nn
import torch
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, in_channels, out_channels, kernel_size, stride=1,
                 dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):

        assert kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride,
                           padding, dilation, groups,
                           bias, padding_mode)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return nn.Conv2d.forward(self, x)

class PixelCNN(nn.Module):

    def __init__(self, fm):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            MaskedConv2d('A', 3,  fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            MaskedConv2d('B', fm, fm, 3, 1, bias=True), nn.BatchNorm2d(fm), nn.LeakyReLU(),
            )

        self.rout: nn.Module = MaskedConv2d('B', fm, 4, 1)
        self.gout: nn.Module = MaskedConv2d('B', fm, 4, 1)
        self.bout: nn.Module = MaskedConv2d('B', fm, 4, 1)


    def forward(self, x):
        y = self.net(x)
        rout = self.rout(y)[:, :, None, ...]
        gout = self.gout(y)[:, :, None, ...]
        bout = self.bout(y)[:, :, None, ...]

        out = torch.cat([rout, gout, bout], dim=-3)
        return out

if __name__ == '__main__':

    # visualize receptive field
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    xin = torch.ones((1, 3, 28, 28), dtype=torch.float)
    xin.requires_grad_(True)
    model: nn.Module = PixelCNN(3)
    model.eval()

    out = model(xin)

    nsample = 0
    category = 1
    in_channel = 0
    out_channel = 2
    position = (14, 14)
    output_target = (nsample, category, out_channel, ) + position
    out[output_target].backward()
    grad = xin.grad[nsample, in_channel].numpy()
    grad[grad != 0] = 1
    plt.imshow(grad, cmap='gray', vmin=0, vmax=1)
    plt.show()
