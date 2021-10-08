import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VAERNNEncoder(nn.Module):
    def __init__(self,
                 # network size
                 layers_before_gru=(),
                 hidden_size=64,
                 layers_after_gru=(),
                 task_inference_latent_dim=32,
                 # actions, states, rewards
                 action_dim=2,
                 action_embed_dim=10,
                 state_dim=2,
                 state_embed_dim=10,
                 reward_size=1,
                 reward_embed_size=5,
                 ):
        super(VAERNNEncoder, self).__init__()

        self.task_inference_latent_dim = task_inference_latent_dim
        self.hidden_size = hidden_size
        self.reparameterise = self._sample_gaussian
        self.state_embed_dim = state_embed_dim

        # embed action, state, reward
        self.state_encoder = utl.SimpleVision(state_embed_dim)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_size, reward_embed_size, F.relu)

        # fully connected layers before the recurrent cell
        curr_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.fc_before_gru = nn.ModuleList([])
        for i in range(len(layers_before_gru)):
            self.fc_before_gru.append(nn.Linear(curr_input_dim, layers_before_gru[i]))
            curr_input_dim = layers_before_gru[i]

        # recurrent unit
        # TODO: TEST RNN vs GRU vs LSTM
        self.gru = nn.GRUCell(input_size=curr_input_dim,
                              hidden_size=hidden_size)

        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        # fully connected layers after the recurrent cell
        curr_input_dim = hidden_size
        self.fc_after_gru = nn.ModuleList([])
        for i in range(len(layers_after_gru)):
            self.fc_after_gru.append(nn.Linear(curr_input_dim, layers_after_gru[i]))
            curr_input_dim = layers_after_gru[i]

        # output layer
        self.fc_mu = nn.Linear(curr_input_dim, task_inference_latent_dim)
        self.fc_logvar = nn.Linear(curr_input_dim, task_inference_latent_dim)

    @staticmethod
    def reset_hidden(hidden_state, done):
        """ Reset the hidden state where the BAMDP was done (i.e., we get a new task) """
        if hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(0)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
        hidden_state = hidden_state * (1 - done)
        return hidden_state

    def prior(self, batch_size, sample=True):

        # TODO: add option to incorporate the initial state

        # we start out with a hidden state of zero
        hidden_state = torch.zeros((1, batch_size, self.hidden_size), requires_grad=True).to(device)

        h = hidden_state
        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            h = F.relu(self.fc_after_gru[i](h))

        # outputs
        latent_mean = self.fc_mu(h)
        latent_logvar = self.fc_logvar(h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, hidden_state

    def forward(self, actions, states, rewards, hidden_state, return_prior, sample=True):
        """
        Actions, states, rewards should be given in form [sequence_len * batch_size * dim].
        For one-step predictions, sequence_len=1 and hidden_state!=None.
        For feeding in entire trajectories, sequence_len>1 and hidden_state=None.
        In the latter case, we return embeddings of length sequence_len+1 since they include the prior.
        """

        # extract features for states, actions, rewards
        ha = self.action_encoder(actions)

        batch_size = states.shape[1]
        seq_len = states.shape[0]
        tmp_state = states.clone()
        states = states.view(-1, states.shape[-1])
        states = utl.image_obs(states)

        hs = self.state_encoder(states)

        tmp1 = hs.view(seq_len, batch_size, self.state_embed_dim - 1)
        tmp2 = tmp_state[:, :, -2:-1]
        hs = torch.cat((tmp1, tmp2), dim=-1)

        hr = self.reward_encoder(rewards)
        h = torch.cat((ha, hs, hr), dim=2)

        # forward through fully connected layers before GRU
        for i in range(len(self.fc_before_gru)):
            h = F.relu(self.fc_before_gru[i](h))

        output = self.gru(h, hidden_state)
        gru_h = output.clone()

        # forward through fully connected layers after GRU
        for i in range(len(self.fc_after_gru)):
            gru_h = F.relu(self.fc_after_gru[i](gru_h))

        # outputs
        latent_mean = self.fc_mu(gru_h)
        latent_logvar = self.fc_logvar(gru_h)
        if sample:
            latent_sample = self.reparameterise(latent_mean, latent_logvar)
        else:
            latent_sample = latent_mean

        return latent_sample, latent_mean, latent_logvar, output
