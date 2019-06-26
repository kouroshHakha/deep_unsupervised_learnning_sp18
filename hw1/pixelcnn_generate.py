import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from pixelcnn_model import PixelCNN

def main(ckt_point_path, nsamples=1, feature_size=128):
    path = Path(ckt_point_path)
    dim = 28
    nchannel = 3
    # Define and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PixelCNN(feature_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    sample = torch.zeros((nsamples, 3, dim, dim)).to(device)

    # Generating images pixel by pixel
    for c in range(nchannel):
        for i in range(dim):
            for j in range(dim):
                out = model(sample)
                probs = F.softmax(out[:, :, c, i, j], dim=-1).data
                sample[:, c, i, j] = probs.multinomial(1).float()
                pdb.set_trace()


    # Saving images row wise
    # torchvision.utils.save_image(sample, 'sample.png', nrow=12, padding=0)


if __name__ == '__main__':
    model_ckt_point = sys.argv[1]
    main(model_ckt_point, 1, feature_size=128)
