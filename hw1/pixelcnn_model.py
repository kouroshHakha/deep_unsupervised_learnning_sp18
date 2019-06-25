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

class ResBlock(nn.Module):

    def __init__(self, fm):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(fm, fm//2, 1, 1, bias=True), nn.LeakyReLU(),
            MaskedConv2d('B', fm//2, fm//2, 3, 1, bias=True), nn.LeakyReLU(),
            nn.Conv2d(fm//2, fm, 1, 1, bias=True), nn.LeakyReLU(),
        )

    def forward(self, x):
        y = self.net(x)
        out = y + x
        return out

class PixelCNN(nn.Module):

    def __init__(self, fm):
        super(PixelCNN, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(3), MaskedConv2d('A', 3,  fm, 3, 1, bias=True), nn.LeakyReLU(),
            nn.BatchNorm2d(fm), ResBlock(fm), # 1
            nn.BatchNorm2d(fm), ResBlock(fm), # 2
            nn.BatchNorm2d(fm), ResBlock(fm), # 3
            nn.BatchNorm2d(fm), ResBlock(fm), # 4
            nn.BatchNorm2d(fm), ResBlock(fm), # 5
            nn.BatchNorm2d(fm), ResBlock(fm), # 6
            nn.BatchNorm2d(fm), ResBlock(fm), # 7
            nn.BatchNorm2d(fm), ResBlock(fm), # 8
            nn.BatchNorm2d(fm), ResBlock(fm), # 9
            nn.BatchNorm2d(fm), ResBlock(fm), # 10
            nn.BatchNorm2d(fm), ResBlock(fm), # 11
            nn.BatchNorm2d(fm), ResBlock(fm), # 12
            nn.BatchNorm2d(fm), nn.Conv2d(fm, fm, 1, 1, bias=True), nn.LeakyReLU(),
            nn.BatchNorm2d(fm),
            )

        self.rout: nn.Module = nn.Conv2d(fm, 4, 1)
        self.gout: nn.Module = nn.Conv2d(fm, 4, 1)
        self.bout: nn.Module = nn.Conv2d(fm, 4, 1)

        self.xentropy =  nn.CrossEntropyLoss(reduction='mean')
        self.out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        rout = self.rout(y)[:, :, None, ...]
        gout = self.gout(y)[:, :, None, ...]
        bout = self.bout(y)[:, :, None, ...]
        self.out = torch.cat([rout, gout, bout], dim=-3)
        return self.out

    def loss(self, target: torch.Tensor) -> torch.Tensor:
        if self.out is None:
            raise ValueError('You should run the model in forward mode at least once')
        return self.xentropy(self.out, target)

class PixelCNNParallel(nn.DataParallel):
    def loss(self, target: torch.Tensor) -> torch.Tensor:
        return self.module.loss(target)

if __name__ == '__main__':

    # visualize receptive field
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    xin = torch.ones((128, 3, 28, 28), dtype=torch.float)
    xin.requires_grad_(True)
    model: nn.Module = PixelCNN(128)
    model.eval()

    pdb.set_trace()
    model = model.to(device)
    xin = xin.to(device)

    out = model(xin)
    pdb.set_trace()

    nsample = 0
    category = 0
    in_channel = 0
    out_channel = 0
    position = (14, 14)
    output_target = (nsample, category, out_channel, ) + position
    out[output_target].backward()
    grad = xin.grad[nsample, in_channel].numpy()
    grad[grad != 0] = 1
    plt.imshow(grad, cmap='gray', vmin=0, vmax=1)
    plt.show()

    pdb.set_trace()