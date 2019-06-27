from typing import cast
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pdb
from pixelcnn_model import PixelCNN
import matplotlib.pyplot as plt
import pickle
import numpy as np

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = torchvision.utils.make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize((512, int(im.size[-1] * 512 // im.size[0])), Image.NEAREST)
    im.save(filename)

def main(ckt_point_path, nsamples=1, feature_size=128):
    path = Path(ckt_point_path)
    images_directory = Path('images')
    images_directory.mkdir(parents=True, exist_ok=True)
    dim = 28
    nchannel = 3
    # Define and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PixelCNN(feature_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    # visualize receptive field
    seed = 20
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
    xin = torch.ones((1, 3, 28, 28), dtype=torch.float, device=device)
    xin.requires_grad_(True)
    model.eval()

    model = model.to(device)

    out = model(xin)

    nsample = 0
    category = 0
    in_channel = 0
    out_channel = 0
    position = (25, 27)
    output_target = (nsample, category, out_channel, ) + position
    out[output_target].backward()
    grad = xin.grad[nsample, in_channel].cpu().numpy()
    # grad[grad != 0] = 1
    grad = np.abs(grad)
    grad /= np.max(grad)
    plt.imshow(grad, cmap='gray', vmin=0, vmax=1)
    plt.savefig(images_directory / 'recf49_25_27.png')

    # sample = torch.zeros((nsamples, 3, dim, dim)).to(device)
    #
    # # Generating images pixel by pixel
    # for i in range(dim):
    #     for j in range(dim):
    #         for c in range(nchannel):
    #             out = model(sample)
    #             probs = F.softmax(out[:, :, c, i, j], dim=-1).data
    #             sample[:, c, i, j] = torch.squeeze(probs.multinomial(1).float() / 4.0)
    #
    # # Saving images row wise
    # save_image(sample, images_directory / 'gen_ckpt5.png', nrow=5, padding=0)


if __name__ == '__main__':
    # np.random.seed(10)
    # dataset_file = './mnist-hw1.pkl'
    # images_directory = Path('./images')
    # images_directory.mkdir(parents=True, exist_ok=True)
    #
    # with open(dataset_file, 'rb') as f:
    #     dataset = pickle.load(f)
    #
    # data = cast(np.ndarray, dataset['train']).transpose([0,3,1,2])
    # samples_ind = np.random.choice(np.arange(len(data)), 100, replace=False)
    # samples = torch.from_numpy(data[samples_ind]).float() / 4.0
    # save_image(samples, images_directory / 'grid_color.png', nrow=5,
    #                              padding=0)

    model_ckt_point = sys.argv[1]
    main(model_ckt_point, 25, feature_size=128)
