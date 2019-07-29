import torch
import torch.nn as nn
import torch.distributions as dist
# import torch.nn.functional as F
import pdb

class Encoder(nn.Module):

    def __init__(self, nx, nz, hidden_layers):
        nn.Module.__init__(self)
        self.nz = nz

        layer_list = [nn.BatchNorm1d(nx)]
        for nin, nout in zip([nx] + hidden_layers[:-1], hidden_layers):
            layer_list += [nn.Linear(nin, nout), nn.BatchNorm1d(nout), nn.ReLU()]
        layer_list += [nn.Linear(hidden_layers[-1], 2 * nz)]
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        out = self.model(x)
        mu, sigma = out[:, :self.nz], out[:, self.nz:]
        sigma = sigma.exp()
        res = torch.zeros(x.shape[0], self.nz, self.nz)
        res.as_strided(sigma.size(), [res.stride(0), res.size(2) + 1]).copy_(sigma)
        zdist = dist.MultivariateNormal(mu, res)
        return zdist

class DecoderA(nn.Module):

    def __init__(self, nz, nx, hidden_layers):
        nn.Module.__init__(self)
        self.nx = nx

        layer_list = [nn.BatchNorm1d(nz)]
        for nin, nout in zip([nz] + hidden_layers[:-1], hidden_layers):
            layer_list += [nn.Linear(nin, nout), nn.BatchNorm1d(nout), nn.ReLU()]
        layer_list += [nn.Linear(hidden_layers[-1], 2 * nx)]
        self.model = nn.Sequential(*layer_list)

    def forward(self, z):
        out = self.model(z)
        mu, log_std = out[:, :self.nx], out[:, self.nx:]
        sigma = log_std.exp()
        res = torch.zeros(z.shape[0], self.nx, self.nx)
        res.as_strided(sigma.size(), [res.stride(0), res.size(2) + 1]).copy_(sigma)
        zdist = dist.MultivariateNormal(mu, res)
        return zdist

class DecoderB(nn.Module):

    def __init__(self, nz, nx, hidden_layers):
        nn.Module.__init__(self)
        self.nx = nx

        layer_list = [nn.BatchNorm1d(nz)]
        for nin, nout in zip([nz] + hidden_layers[:-1], hidden_layers):
            layer_list += [nn.Linear(nin, nout), nn.BatchNorm1d(nout), nn.ReLU()]
        layer_list += [nn.Linear(hidden_layers[-1], nx + 1)]
        self.model = nn.Sequential(*layer_list)

    def forward(self, z):
        out = self.model(z)
        mu, sigma = out[:, :self.nx], out[:, -1]
        sigma = sigma.exp()
        cov_mat = sigma[:, None, None] * torch.eye(self.nx, self.nx)
        zdist = dist.MultivariateNormal(mu, cov_mat)
        return zdist

class VAE2D(nn.Module):

    def __init__(self, nx, nz, hidden_layers, question='A'):
        nn.Module.__init__(self)

        self.nz = nz
        self.encoder: nn.Module = Encoder(nx, nz, hidden_layers)
        self.prior = dist.MultivariateNormal(torch.zeros(self.nz), torch.eye(self.nz))
        if question == 'A':
            self.decoder: nn.Module  = DecoderA(nz, nx, hidden_layers)
        else:
            self.decoder: nn.Module = DecoderB(nz, nx, hidden_layers)

    def forward(self, x, analytic_kl=True):
        posterior: dist.MultivariateNormal = self.encoder(x)
        mu, var = posterior.mean, posterior.variance
        eps_dist = dist.MultivariateNormal(torch.zeros(self.nz), torch.eye(self.nz))
        torch.manual_seed(7)
        eps = eps_dist.sample((x.shape[0], ))
        z = var * eps + mu
        x_dist: dist.MultivariateNormal = self.decoder(z)
        xhat = x_dist.sample()
        nll_recon = -x_dist.log_prob(x).mean(dim=-1)
        if analytic_kl:
            kl = dist.kl_divergence(posterior, self.prior)
        else:
            kl = posterior.log_prob(z) - self.prior.log_prob(z)

        kl_pp = kl.mean(dim=-1)
        loss = nll_recon + kl_pp
        return loss, nll_recon, kl_pp, z, xhat, x_dist, kl


if __name__ == '__main__':
    import hw3.dataset as dt
    samples = dt.sample_data_1()
    samples = torch.from_numpy(samples).float()
    # encoder: nn.Module = Encoder(2, 2, hidden_layers=[40, 20, 20])
    # encoder.eval()
    # zs, mu, sigma = encoder(samples)

    vae: nn.Module = VAE2D(2,2,[20, 20, 20])
    vae.eval()
    loss, z, xhat = vae(samples[:10])
    import pdb
    pdb.set_trace()