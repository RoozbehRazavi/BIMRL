import torch
import numpy as np
import torch.nn as nn
from collections import OrderedDict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class MLP(nn.Module):
    def __init__(self, hidden_size, last_activation=True):
        super(MLP, self).__init__()
        q = []
        for i in range(len(hidden_size) - 1):
            in_dim = hidden_size[i]
            out_dim = hidden_size[i + 1]
            q.append(("Linear_%d" % i, nn.Linear(in_dim, out_dim)))
            if (i < len(hidden_size) - 2) or ((i == len(hidden_size) - 2) and (last_activation)):
                q.append(("BatchNorm_%d" % i, nn.BatchNorm1d(out_dim)))
                q.append(("ReLU_%d" % i, nn.ReLU(inplace=True)))
        self.mlp = nn.Sequential(OrderedDict(q))

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, obs_dim, brim_hidden_dim, nhid=16):
        super(Encoder, self).__init__()
        self.obs_encoder = nn.Linear(obs_dim, nhid)
        self.brim_hidden_encoder = nn.Linear(brim_hidden_dim, nhid)
        self.calc_mean = nn.Linear(nhid * 2, nhid)
        self.calc_logvar = nn.Linear(nhid * 2, nhid)

    def forward(self, brim_hidden, obs=None):
        obs = self.obs_encoder(obs)
        brim_hidden = self.brim_hidden_encoder(brim_hidden)

        return self.calc_mean(torch.cat((obs, brim_hidden), dim=1)), self.calc_logvar(
            torch.cat((obs, brim_hidden), dim=1))


class Decoder(nn.Module):
    def __init__(self, brim_hidden_dim, obs_dim, nhid=16):
        super(Decoder, self).__init__()
        self.decoder = nn.Linear(nhid + obs_dim, brim_hidden_dim)

    def forward(self, z, obs=None):
        return self.decoder(torch.cat((z, obs), dim=-1))


class cVAE(nn.Module):

    def __init__(self, obs_dim, brim_hidden_dim, nhid=16):
        super(cVAE, self).__init__()
        self.encoder = Encoder(obs_dim, brim_hidden_dim, nhid)
        self.decoder = Decoder(brim_hidden_dim, obs_dim, nhid)
        self.MSELoss = nn.MSELoss(reduction="mean")
        self.dim = nhid

    def sampling(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        sigma = 0.5 * torch.exp(logvar)

        return mean + eps * sigma

    def loss(self, X, X_hat, mean, logvar):
        reconstruction_loss = self.MSELoss(X_hat, X)
        KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)

        return reconstruction_loss + KL_divergence

    def forward(self, brim_hidden_state, obs):
        mean, logvar = self.encoder(brim_hidden_state, obs)
        z = self.sampling(mean, logvar)
        return self.decoder(z, obs), mean, logvar

    def generate(self, obs):
        obs = obs.to(device)
        if (len(obs.shape) == 0):
            batch_size = None
            obs = obs.unsqueeze(0)
            z = torch.randn((1, self.dim)).to(device)
        else:
            batch_size = obs.shape[0]
            z = torch.randn((batch_size, self.dim)).to(device)
        res = self.decoder(z, obs)
        if not batch_size:
            res = res.squeeze(0)
        return res
