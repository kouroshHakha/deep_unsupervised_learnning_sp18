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
    # Define and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PixelCNN(feature_size).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()

    sample = torch.zeros((nsamples, 3, dim, dim)).to(device)

    # Generating images pixel by pixel
    for i in range(dim):
        for j in range(dim):
            out = model(sample)
            pdb.set_trace()
            probs = F.softmax(out[:, :, i, j], dim=-1).data
            sample[:, :, i, j] = torch.multinomial(probs, 1).float() / 255.0

    # Saving images row wise
    # torchvision.utils.save_image(sample, 'sample.png', nrow=12, padding=0)


if __name__ == '__main__':
    model_ckt_point = sys.argv[1]
    main(model_ckt_point, 1, feature_size=128)
