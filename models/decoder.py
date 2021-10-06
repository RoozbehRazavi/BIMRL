import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import time
class StateTransitionDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 action_simulator_hidden_size,
                 pred_type='deterministic',
                 n_step_state_prediction=True,
                 n_prediction=3
                 ):
        super(StateTransitionDecoder, self).__init__()
        self.n_step_state_prediction = n_step_state_prediction
        self.n_prediction = n_prediction

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if n_step_state_prediction:
            # RNN for simulate future state base on current state and future actions
            self.h_to_hidden_state = nn.Sequential(
                nn.Linear(curr_input_dim, action_simulator_hidden_size*2),
                nn.ReLU(),
                nn.Linear(action_simulator_hidden_size*2, action_simulator_hidden_size)
            )
            self.action_simulator = nn.GRUCell(action_embed_dim, action_simulator_hidden_size)
            self.n_step_fc_out = nn.ModuleList([])
            for i in range(self.n_prediction):
                if pred_type == 'gaussian':
                    self.n_step_fc_out.append(nn.Linear(action_simulator_hidden_size, 2 * state_dim))
                else:
                    self.n_step_fc_out.append(nn.Linear(action_simulator_hidden_size, state_dim))
            self.n_step_action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        # output layer
        if pred_type == 'gaussian':
            self.one_step_fc_out = nn.Linear(curr_input_dim, 2 * state_dim)
        else:
            self.one_step_fc_out = nn.Linear(curr_input_dim, state_dim)

    def forward(self, latent_state, state, action, dec_n_step_action):
        assert dec_n_step_action is not None or not self.n_step_state_prediction
        state_prediction = []

        ha = self.action_encoder(action)
        hs = self.state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        state_prediction.append(self.one_step_fc_out(h))

        if self.n_step_state_prediction:
            h = self.h_to_hidden_state(h)
            for i in range(self.n_prediction):
                ha = self.n_step_action_encoder(dec_n_step_action[i])
                ha = ha.reshape((-1, ha.shape[-1]))
                h_size = h.shape
                h = h.reshape((-1, h.shape[-1]))
                h = self.action_simulator(ha, h)
                h = h.reshape((*h_size[:-1], h.shape[-1]))
                state_prediction.append(self.n_step_fc_out[i](h))
        return state_prediction


class RewardDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 ):
        super(RewardDecoder, self).__init__()

        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        if self.multi_head:
            # one output head per state to predict rewards
            curr_input_dim = latent_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]
            self.fc_out = nn.Linear(curr_input_dim, num_states)
        else:
            # get state as input and predict reward prob
            self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
            self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
            curr_input_dim = latent_dim + state_embed_dim
            if input_prev_state:
                curr_input_dim += state_embed_dim
            if input_action:
                curr_input_dim += action_embed_dim
            self.fc_layers = nn.ModuleList([])
            for i in range(len(layers)):
                self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
                curr_input_dim = layers[i]

            if pred_type == 'gaussian':
                self.fc_out = nn.Linear(curr_input_dim, 2)
            else:
                self.fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, next_state, prev_state=None, action=None):

        if self.multi_head:
            h = latent_state.clone()
        else:
            hns = self.state_encoder(next_state)
            h = torch.cat((latent_state, hns), dim=-1)
            if self.input_action:
                ha = self.action_encoder(action)
                h = torch.cat((h, ha), dim=-1)
            if self.input_prev_state:
                hps = self.state_encoder(prev_state)
                h = torch.cat((h, hps), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)


class TaskDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 pred_type,
                 task_dim,
                 num_tasks,
                 ):
        super(TaskDecoder, self).__init__()

        # "task_description" or "task id"
        self.pred_type = pred_type

        curr_input_dim = latent_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        output_dim = task_dim if pred_type == 'task_description' else num_tasks
        self.fc_out = nn.Linear(curr_input_dim, output_dim)

    def forward(self, latent_state):

        h = latent_state

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))

        return self.fc_out(h)
