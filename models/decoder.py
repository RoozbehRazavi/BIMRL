import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ValueDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 value_simulator_hidden_size,
                 pred_type='deterministic',
                 n_prediction=3
                 ):
        super(ValueDecoder, self).__init__()
        self.n_prediction = n_prediction

        self.state_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + action_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        self.h_to_hidden_state = nn.Sequential(
            nn.Linear(curr_input_dim, curr_input_dim*2),
            nn.ReLU(),
            nn.Linear(curr_input_dim*2, curr_input_dim)
        )
        self.value_simulator = nn.GRUCell(action_embed_dim, curr_input_dim)
        # self.n_step_fc_out = nn.ModuleList([])
        # for i in range(self.n_prediction):
        #     if pred_type == 'gaussian':
        #         self.n_step_fc_out.append(nn.Linear(curr_input_dim, 2))
        #     else:
        #         self.n_step_fc_out.append(nn.Linear(curr_input_dim, 1))
        # self.n_step_action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)

        # output layer
        if pred_type == 'gaussian':
            self.one_step_fc_out = nn.Linear(curr_input_dim, 2)
        else:
            self.one_step_fc_out = nn.Linear(curr_input_dim, 1)

    def forward(self, latent_state, state, rewards, action, n_step_action, n_step_rewards):
        value_prediction = []

        # one step value prediction
        ha = self.action_encoder(action)
        hs = self.state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        value_prediction.append(self.one_step_fc_out(h))

        #n step value prediction
        h = self.h_to_hidden_state(h)
        for i in range(self.n_prediction):
            ha = self.action_encoder(n_step_action[i])
            ha = ha.reshape((-1, ha.shape[-1]))
            h_size = h.shape
            h = h.reshape((-1, h.shape[-1]))
            h = self.value_simulator(ha, h)
            h = h.reshape((*h_size[:-1], h.shape[-1]))
            value_prediction.append(self.one_step_fc_out(h))
        return value_prediction


class ActionDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 state_dim,
                 state_embed_dim,
                 state_simulator_hidden_size,
                 action_space,
                 pred_type='deterministic',
                 n_step_action_prediction=True,
                 n_prediction=3
                 ):
        super(ActionDecoder, self).__init__()
        self.n_step_action_prediction = n_step_action_prediction
        self.n_prediction = n_prediction
        self.action_log_dim = action_space.n

        self.state_t_encoder = utl.FeatureExtractor(state_dim, state_embed_dim, F.relu)

        curr_input_dim = latent_dim + state_embed_dim + state_embed_dim
        self.fc_layers = nn.ModuleList([])
        for i in range(len(layers)):
            self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
            curr_input_dim = layers[i]

        if n_step_action_prediction:
            # RNN for simulate future state base on current state and future actions
            self.h_to_hidden_state = nn.Sequential(
                nn.Linear(curr_input_dim, curr_input_dim*2),
                nn.ReLU(),
                nn.Linear(curr_input_dim*2, curr_input_dim)
            )
            self.action_simulator = nn.GRUCell(state_embed_dim, curr_input_dim)

        # output layer
        if pred_type == 'gaussian':
            self.one_step_fc_out = nn.Linear(curr_input_dim, 2*self.action_log_dim)
        else:
            self.one_step_fc_out = nn.Linear(curr_input_dim, self.action_log_dim)
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self,
                latent_state,
                state,
                next_state,
                n_step_next_state,
                n_step_action_prediction,
                ):
        action_prediction = []

        hs_t = self.state_t_encoder(state)
        hs_t_1 = self.state_t_encoder(next_state)
        h = torch.cat((latent_state, hs_t, hs_t_1), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        action_prediction.append(self.log_softmax(self.one_step_fc_out(h)))

        if n_step_action_prediction:
            h = self.h_to_hidden_state(h)
            for i in range(self.n_prediction):
                hs_t_1 = self.state_t_encoder(n_step_next_state[i])
                hs_t_1 = hs_t_1.reshape((-1, hs_t_1.shape[-1]))
                h_size = h.shape
                h = h.reshape((-1, h.shape[-1]))
                h = self.action_simulator(hs_t_1, h)
                h = h.reshape((*h_size[:-1], h.shape[-1]))
                action_prediction.append(self.log_softmax(self.one_step_fc_out(h)))
        return action_prediction


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
                 n_prediction=3,
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
                nn.Linear(curr_input_dim, curr_input_dim*2),
                nn.ReLU(),
                nn.Linear(curr_input_dim*2, curr_input_dim)
            )
            self.action_simulator = nn.GRUCell(action_embed_dim, curr_input_dim)

        # output layer
        if pred_type == 'gaussian':
            self.one_step_fc_out = nn.Linear(curr_input_dim, 2 * state_embed_dim)
        else:
            self.one_step_fc_out = nn.Linear(curr_input_dim, state_embed_dim)

    def forward(self, latent_state, state, action, n_step_action, n_step_state_prediction, state_encoder=None):
        assert n_step_action is not None or not n_step_state_prediction
        assert state_encoder is not None
        state_prediction = []

        ha = self.action_encoder(action)
        hs = self.state_encoder(state) if state_encoder is None else state_encoder(state)
        h = torch.cat((latent_state, hs, ha), dim=-1)

        for i in range(len(self.fc_layers)):
            h = F.relu(self.fc_layers[i](h))
        state_prediction.append(self.one_step_fc_out(h))

        if n_step_state_prediction:
            h = self.h_to_hidden_state(h)
            for i in range(self.n_prediction):
                ha = self.action_encoder(n_step_action[i])
                ha = ha.reshape((-1, ha.shape[-1]))
                h_size = h.shape
                h = h.reshape((-1, h.shape[-1]))
                h = self.action_simulator(ha, h)
                h = h.reshape((*h_size[:-1], h.shape[-1]))
                state_prediction.append(self.one_step_fc_out(h))
        return state_prediction


class RewardDecoder(nn.Module):
    def __init__(self,
                 layers,
                 latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 reward_simulator_hidden_size,
                 num_states,
                 multi_head=False,
                 pred_type='deterministic',
                 input_prev_state=True,
                 input_action=True,
                 n_step_reward_prediction=True,
                 n_prediction=3
                 ):
        super(RewardDecoder, self).__init__()

        self.n_step_reward_prediction = n_step_reward_prediction
        self.n_prediction = n_prediction
        self.pred_type = pred_type
        self.multi_head = multi_head
        self.input_prev_state = input_prev_state
        self.input_action = input_action

        # if self.multi_head:
        #     # one output head per state to predict rewards
        #     curr_input_dim = latent_dim
        #     self.fc_layers = nn.ModuleList([])
        #     for i in range(len(layers)):
        #         self.fc_layers.append(nn.Linear(curr_input_dim, layers[i]))
        #         curr_input_dim = layers[i]
        #     self.fc_out = nn.Linear(curr_input_dim, num_states)
        # else:

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
            self.one_step_fc_out = nn.Linear(curr_input_dim, 2)
        else:
            self.one_step_fc_out = nn.Linear(curr_input_dim, 1)

        if self.n_step_reward_prediction:
            self.h_to_hidden_state = nn.Sequential(
                nn.Linear(curr_input_dim, curr_input_dim * 2),
                nn.ReLU(),
                nn.Linear(curr_input_dim * 2, curr_input_dim))

            gru_input_dim = state_embed_dim
            if input_prev_state:
                gru_input_dim += state_embed_dim
            if input_action:
                gru_input_dim += action_embed_dim
            self.reward_simulator = nn.GRUCell(gru_input_dim, curr_input_dim)

    def forward(self, latent_state, next_state, prev_state=None, action=None, n_step_next_obs=None, n_step_actions=None, n_step_reward_prediction=None):
        if n_step_reward_prediction is None:
            n_step_reward_prediction = self.n_step_reward_prediction
        assert (n_step_next_obs is not None and n_step_actions is not None) or not n_step_reward_prediction

        reward_prediction = []
        # if self.multi_head:
        #     h = latent_state.clone()
        # else:

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

        reward_prediction.append(self.one_step_fc_out(h))

        if n_step_reward_prediction:
            h = self.h_to_hidden_state(h)
            for i in range(self.n_prediction):
                nhs = self.state_encoder(n_step_next_obs[i])
                if self.input_action:
                    ha = self.action_encoder(n_step_actions[i])
                else:
                    ha = torch.zeros(size=(0, ))
                if self.input_prev_state:
                    if i == 0:
                        hps = self.state_encoder(prev_state)
                    else:
                        hps = self.state_encoder(n_step_next_obs[i-1])
                else:
                    hps = torch.zeros(size=(0,), device=device)

                hr = torch.cat((nhs, ha, hps), dim=-1)

                hr = hr.reshape((-1, hr.shape[-1]))
                h_size = h.shape
                h = h.reshape((-1, h.shape[-1]))
                h = self.reward_simulator(hr, h)
                h = h.reshape((*h_size[:-1], h.shape[-1]))
                reward_prediction.append(self.one_step_fc_out(h))
        return reward_prediction


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
