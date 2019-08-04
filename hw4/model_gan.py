import torch
import torch.nn as nn


class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        # NCHW
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        q, r = divmod(d_depth, self.block_size_sq)
        if r != 0:
            raise ValueError(f'Depth {d_depth} is not divisable by bs x bs {self.block_size_sq}')
        s_depth = int(q)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1)
        output = output.permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class SpaceToDepth(nn.Module):
    def __init__(self, block_size):
        super(SpaceToDepth, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        # NCHW
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()

        d_width, rw = divmod(s_width, self.block_size)
        d_height, rh = divmod(s_height, self.block_size)
        if rw != 0  or rh != 0:
            raise ValueError(f'size is not divisable by block_size = {self.block_size}')
        d_depth = s_depth * self.block_size_sq
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.contiguous().view(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class UpsampleConv2D(nn.Module):
    # NCHW
    def __init__(self, input_channels, filter_size, n_filters, **kwargs):
        nn.Module.__init__(self)
        self.conv2d = nn.Conv2d(input_channels, n_filters, filter_size, **kwargs)
        self.d2sp: nn.Module = DepthToSpace(block_size=2)

    def forward(self, x):
        x0 = torch.cat([x for _ in range(4)], dim=1)
        x1 = self.d2sp(x0)
        x2 = self.conv2d(x1)
        return x2

class DownsampleConv2D(nn.Module):
    # NCHW
    def __init__(self, input_channels, filter_size, n_filters, **kwargs):
        nn.Module.__init__(self)
        self.conv2d = nn.Conv2d(input_channels, n_filters, filter_size, **kwargs)
        self.sp2d: nn.Module = SpaceToDepth(block_size=2)

    def forward(self, x):
        x0 = self.sp2d(x)
        sp_size = int(x0.shape[1] / 4)
        x1 = sum(torch.split(x0, sp_size, dim=1)) / 4
        x2 = self.conv2d(x1)
        return x2

class ResBlockUp(nn.Module):
    def __init__(self, input_channels, filter_size, n_filters):
        nn.Module.__init__(self)

        padding = tuple([(x - 1) // 2 for x in filter_size])
        forward_pass_layers = [nn.BatchNorm2d(input_channels),
                               nn.ReLU(),
                               nn.Conv2d(input_channels, input_channels, filter_size,
                                         padding=padding),
                               nn.BatchNorm2d(input_channels),
                               nn.ReLU(),
                               UpsampleConv2D(input_channels, filter_size, n_filters,
                                              padding=padding)]
        self.forward_pass = nn.Sequential(*forward_pass_layers)
        self.shorcut: nn.Module = UpsampleConv2D(input_channels, (1,1), n_filters, padding=0)

    def forward(self, x):
        residual = self.forward_pass(x)
        shortcut = self.shorcut(x)
        y = residual + shortcut
        return y


class ResBlockDown(nn.Module):
    def __init__(self, input_channels, filter_size, n_filters):
        nn.Module.__init__(self)

        padding = tuple([(x - 1) // 2 for x in filter_size])
        forward_pass_layers = [nn.BatchNorm2d(input_channels),
                               nn.ReLU(),
                               nn.Conv2d(input_channels, input_channels, filter_size,
                                         padding=padding),
                               nn.BatchNorm2d(input_channels),
                               nn.ReLU(),
                               DownsampleConv2D(input_channels, filter_size, n_filters,
                                              padding=padding)]
        self.forward_pass = nn.Sequential(*forward_pass_layers)
        self.shorcut: nn.Module = DownsampleConv2D(input_channels, (1,1), n_filters, padding=0)


    def forward(self, x):
            residual = self.forward_pass(x)
            shortcut = self.shorcut(x)
            y = residual + shortcut
            return y

class ResBlock(nn.Module):
    def __init__(self, input_channels, filter_size, n_filters):
        nn.Module.__init__(self)
        padding = tuple([(x - 1) // 2 for x in filter_size])
        forward_pass_layers = [nn.ReLU(),
                               nn.Conv2d(input_channels, input_channels, filter_size,
                                         padding=padding),
                               nn.ReLU(),
                               nn.Conv2d(input_channels, n_filters, filter_size,
                                         padding=padding)]
        self.forward_pass = nn.Sequential(*forward_pass_layers)

    def forward(self, x):
        residual = self.forward_pass(x)
        y = residual + x
        return y

class Generator(nn.Module):
    # NCHW
    def __init__(self, nz):
        nn.Module.__init__(self)

        self.input_layer = nn.Linear(nz, 4 * 4 * 256)
        res_block_list = [ResBlockUp(256, (3,3), 256),
                          ResBlockUp(256, (3,3), 256),
                          ResBlockUp(256, (3,3), 256)]
        self.res_nets = nn.Sequential(*res_block_list)
        output_layers = [nn.BatchNorm2d(256),
                         nn.ReLU(),
                         nn.Conv2d(256, 3, (3,3), padding=1),
                         nn.Tanh()]
        self.output_nets = nn.Sequential(*output_layers)

    def forward(self, x):
        gen_x1 = self.input_layer(x).view(-1, 256, 4, 4)
        gen_x2 = self.res_nets(gen_x1)
        gen_x = self.output_nets(gen_x2)

        return gen_x

class Discriminator(nn.Module):
    # NCHW
    def __init__(self):
        nn.Module.__init__(self)

        res_block_list = [ResBlockDown(3, (3,3), 128),
                          ResBlockDown(128, (3,3), 128),
                          ResBlock(128, (3,3), 128),
                          ResBlock(128, (3,3), 128),
                          nn.ReLU(),
                          nn.AvgPool2d((8,8))]
        self.res_nets = nn.Sequential(*res_block_list)
        self.output_nets = nn.Linear(128, 1)

    def forward(self, x):
        disc_x1 = self.res_nets(x).squeeze(dim=-1).squeeze(dim=-1)
        logits = self.output_nets(disc_x1)
        # y = logits.sigmoid()
        return logits

class WGANModel(nn.Module):

    def __init__(self, nz):
        nn.Module.__init__(self)
        self.gen: nn.Module = Generator(nz)
        self.disc: nn.Module = Discriminator()

    def forward(self, x):
        gen_x = self.gen(x)
        logits = self.disc(gen_x)
        return logits, gen_x

    def generate(self, x):
        return self.gen(x)

    def discriminate(self, input) -> torch.Tensor:
        return self.disc(input)

if __name__ == '__main__':

    x = torch.randn(1000, 128)
    gen: nn.Module = Generator(nz=128)
    disc: nn.Module = Discriminator()
    gen.eval()
    disc.eval()
    y = gen(x)
    print(f'x_shape = {x.shape}')
    print(f'y_shape = {y.shape}')
    z = disc(y)
    print(f'z_shape = {z.shape}')
    import pdb
    pdb.set_trace()
