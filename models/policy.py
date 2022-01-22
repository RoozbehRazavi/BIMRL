"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import numpy as np
import torch
import torch.nn as nn

from utils import helpers as utl
from visual_attention import VisionCore

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module):
    def __init__(self,
                 args,
                 # input
                 pass_state_to_policy,
                 pass_task_inference_latent_to_policy,
                 pass_belief_to_policy,
                 pass_task_to_policy,
                 pass_rim_level1_output_to_policy,
                 dim_state,
                 task_inference_latent_dim,
                 rim_level1_output_dim,
                 dim_belief,
                 dim_task,
                 # hidden
                 hidden_layers,
                 activation_function,  # tanh, relu, leaky-relu
                 policy_initialisation,  # orthogonal / normc
                 # output
                 action_space,
                 init_std,
                 norm_actions_of_policy,
                 action_low,
                 action_high,
                 ):
        """
        The policy can get any of these as input:
        - state (given by environment)
        - task (in the (belief) oracle setting)
        - latent variable (from VAE)
        """
        super(Policy, self).__init__()

        self.args = args

        # Stochastic behavior remove for policy
        task_inference_latent_dim *= 2

        if activation_function == 'tanh':
            self.activation_function = nn.Tanh()
        elif activation_function == 'relu':
            self.activation_function = nn.ReLU()
        elif activation_function == 'leaky-relu':
            self.activation_function = nn.LeakyReLU()
        else:
            raise ValueError

        if policy_initialisation == 'normc':
            init_ = lambda m: init(m, init_normc_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(activation_function))
        elif policy_initialisation == 'orthogonal':
            init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain(activation_function))

        self.pass_state_to_policy = pass_state_to_policy
        self.pass_task_inference_latent_to_policy = pass_task_inference_latent_to_policy
        self.pass_task_to_policy = pass_task_to_policy
        self.pass_belief_to_policy = pass_belief_to_policy
        self.pass_rim_level1_output_to_policy = pass_rim_level1_output_to_policy

        # set normalisation parameters for the inputs
        # (will be updated from outside using the RL batches)
        self.norm_rim_level1_output = self.args.norm_rim_level1_output and (rim_level1_output_dim is not None)
        if self.pass_rim_level1_output_to_policy and self.norm_rim_level1_output:
            self.rim_level1_output_rms = utl.RunningMeanStd(shape=(rim_level1_output_dim))

        self.norm_state = self.args.norm_state_for_policy and (dim_state is not None)
        if self.pass_state_to_policy and self.norm_state:
            self.state_rms = utl.RunningMeanStd(shape=(dim_state))

        self.norm_task_inference_latent = self.args.norm_task_inference_latent_for_policy and (task_inference_latent_dim is not None)
        if self.pass_task_inference_latent_to_policy and self.norm_task_inference_latent:
            self.task_inference_latent_rms = utl.RunningMeanStd(shape=(task_inference_latent_dim))

        self.norm_belief = self.args.norm_belief_for_policy and (dim_task is not None)
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms = utl.RunningMeanStd(shape=(dim_belief))

        self.norm_task = self.args.norm_task_for_policy and (dim_belief is not None)
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms = utl.RunningMeanStd(shape=(dim_task))

        curr_input_dim = dim_state * int(self.pass_state_to_policy) + \
                         task_inference_latent_dim * int(self.pass_task_inference_latent_to_policy) + \
                         dim_belief * int(self.pass_belief_to_policy) + \
                         dim_task * int(self.pass_task_to_policy) +\
                         rim_level1_output_dim * int(self.pass_rim_level1_output_to_policy)

        # initialise encoders for separate inputs
        self.use_state_encoder = self.args.policy_state_embedding_dim is not None
        if self.pass_state_to_policy and self.use_state_encoder:
            if self.args.use_stateful_vision_core:
                assert self.args.use_rim_level1
                self.state_encoder = VisionCore(
                    hidden_size=self.args.rim_output_size_to_vision_core,
                    c_v=self.args.visual_attention_value_size,
                    c_k=self.args.visual_attention_key_size,
                    c_s=self.args.visual_attention_spatial,
                    num_queries=self.args.visual_attention_num_queries,
                    state_embedding_size=self.args.policy_state_embedding_dim)

            else:
                self.state_encoder = utl.SimpleVision(self.args.policy_state_embedding_dim)
            curr_input_dim = curr_input_dim - dim_state + self.args.policy_state_embedding_dim

        self.use_task_inference_latent_encoder = self.args.policy_task_inference_latent_embedding_dim is not None
        if self.pass_task_inference_latent_to_policy and self.use_task_inference_latent_encoder:
            self.task_inference_latent_encoder = utl.FeatureExtractor(task_inference_latent_dim, self.args.policy_task_inference_latent_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - task_inference_latent_dim + self.args.policy_task_inference_latent_embedding_dim

        self.use_belief_encoder = self.args.policy_belief_embedding_dim is not None
        if self.pass_belief_to_policy and self.use_belief_encoder:
            self.belief_encoder = utl.FeatureExtractor(dim_belief, self.args.policy_belief_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_belief + self.args.policy_belief_embedding_dim

        self.use_task_encoder = self.args.policy_task_embedding_dim is not None
        if self.pass_task_to_policy and self.use_task_encoder:
            self.task_encoder = utl.FeatureExtractor(dim_task, self.args.policy_task_embedding_dim, self.activation_function)
            curr_input_dim = curr_input_dim - dim_task + self.args.policy_task_embedding_dim

        self.use_rim_level1_output_encoder = self.args.policy_rim_level1_output_embedding_dim is not None
        if self.pass_rim_level1_output_to_policy and self.use_rim_level1_output_encoder:
            self.rim_level1_output_encoder = utl.FeatureExtractor(task_inference_latent_dim, self.args.policy_latent_embedding_dim,
                                                       self.activation_function)
            curr_input_dim = curr_input_dim - rim_level1_output_dim + self.args.policy_rim_level1_output_embedding_dim

        # initialise actor and critic
        hidden_layers = [int(h) for h in hidden_layers]
        self.actor_layers = nn.ModuleList()
        self.critic_layers = nn.ModuleList()
        for i in range(len(hidden_layers)):
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.actor_layers.append(fc)
            fc = init_(nn.Linear(curr_input_dim, hidden_layers[i]))
            self.critic_layers.append(fc)
            curr_input_dim = hidden_layers[i]
        self.critic_linear = nn.Linear(hidden_layers[-1], 1)

        # output distributions of the policy
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(hidden_layers[-1], num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(hidden_layers[-1], num_outputs, init_std, min_std=1e-6,
                                     action_low=action_low, action_high=action_high,
                                     norm_actions_of_policy=norm_actions_of_policy)
        else:
            raise NotImplementedError

    def get_actor_params(self):
        return [*self.actor.parameters(), *self.dist.parameters()]

    def get_critic_params(self):
        return [*self.critic.parameters(), *self.critic_linear.parameters()]

    def forward_actor(self, inputs):
        h = inputs
        for i in range(len(self.actor_layers)):
            h = self.actor_layers[i](h)
            h = self.activation_function(h)
        return h

    def forward_critic(self, inputs):
        h = inputs
        for i in range(len(self.critic_layers)):
            h = self.critic_layers[i](h)
            h = self.activation_function(h)
        return h

    def state_process(self, state, rim_output_to_vision_core=None):
        if self.pass_state_to_policy:
            if self.norm_state:
                state = (state - self.state_rms.mean) / torch.sqrt(self.state_rms.var + 1e-8)
            if self.use_state_encoder:
                batch_size = state.shape[0]
                tmp_state = state.clone()
                state = state.view(-1, state.shape[-1])
                state = utl.image_obs(state)

                if self.args.use_stateful_vision_core:
                    hs = self.state_encoder(state, rim_output_to_vision_core)
                else:
                    hs = self.state_encoder(state)

                tmp1 = hs.view(batch_size, self.args.policy_state_embedding_dim - 1)
                tmp2 = tmp_state[:, -2:-1]
                state = torch.cat((tmp1, tmp2), dim=-1)
        else:
            state = torch.zeros(0, ).to(device)
        return state

    def forward(self, embedded_state, task_inference_latent, brim_output_level1, belief, task):
        state = embedded_state
        if self.pass_task_inference_latent_to_policy:
            if self.norm_task_inference_latent:
                task_inference_latent = (task_inference_latent - self.task_inference_latent_rms.mean) / torch.sqrt(self.task_inference_latent_rms.var + 1e-8)
            if self.use_task_inference_latent_encoder:
                task_inference_latent = self.task_inference_latent_encoder(task_inference_latent)
            if len(task_inference_latent.shape) == 1 and len(state.shape) == 2:
                task_inference_latent = task_inference_latent.unsqueeze(0)
        else:
            task_inference_latent = torch.zeros(0, ).to(device)

        if self.pass_rim_level1_output_to_policy:
            if self.norm_rim_level1_output:
                brim_output_level1 = (brim_output_level1 - self.rim_level1_output_rms.mean) / torch.sqrt(self.rim_level1_output_rms.var + 1e-8)
            if self.use_rim_level1_output_encoder:
                brim_output_level1 = self.rim_level1_output_encoder(brim_output_level1)
            if len(brim_output_level1.shape) == 1 and len(state.shape) == 2:
                brim_output_level1 = brim_output_level1.unsqueeze(0)
        else:
            brim_output_level1 = torch.zeros(0, ).to(device)

        if self.pass_belief_to_policy:
            if self.norm_belief:
                belief = (belief - self.belief_rms.mean) / torch.sqrt(self.belief_rms.var + 1e-8)
            if self.use_belief_encoder:
                belief = self.belief_encoder(belief)
            belief = belief.float()
            if len(belief.shape) == 1 and len(state.shape) == 2:
                belief = belief.unsqueeze(0)
        else:
            belief = torch.zeros(0, ).to(device)

        if self.pass_task_to_policy:
            if self.norm_task:
                task = (task - self.task_rms.mean) / torch.sqrt(self.task_rms.var + 1e-8)
            if self.use_task_encoder:
                task = self.task_encoder(task.float())
            if len(task.shape) == 1 and len(state.shape) == 2:
                task = task.unsqueeze(0)
        else:
            task = torch.zeros(0, ).to(device)

        # concatenate inputs
        if brim_output_level1.dim() == 3:
            brim_output_level1 = brim_output_level1.squeeze(0)
        if not self.args.rl_loss_throughout_vae_encoder:
            task_inference_latent = task_inference_latent.detach()
        inputs = torch.cat((state, task_inference_latent, brim_output_level1, belief, task), dim=-1)

        # forward through critic/actor part
        hidden_critic = self.forward_critic(inputs)
        hidden_actor = self.forward_actor(inputs)
        return self.critic_linear(hidden_critic), hidden_actor

    def act(self, embedded_state, latent, brim_output_level1, belief, task, deterministic=False):
        value, actor_features = self.forward(embedded_state=embedded_state, task_inference_latent=latent, brim_output_level1=brim_output_level1, belief=belief, task=task)
        try:
            dist = self.dist(actor_features)
        except:
            print('brim_output_level1: ', brim_output_level1)
            import traceback
            traceback.print_exc()
            exit()
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def get_value(self, embedded_state, latent, brim_output_level1, belief, task):
        value, _ = self.forward(embedded_state, latent, brim_output_level1, belief, task)
        return value

    def update_rms(self, args, policy_storage):
        """ Update normalisation parameters for inputs with current data """
        if self.pass_state_to_policy and self.norm_state:
            state = policy_storage.prev_state[:-1]
            self.state_rms.update(state)
        if self.pass_task_inference_latent_to_policy and self.norm_task_inference_latent:
            task_inference_latent = utl.get_latent_for_policy(args.sample_embeddings,
                                                              args.add_nonlinearity_to_latent,
                                                              torch.cat(policy_storage.latent_samples[:-1]),
                                                              torch.cat(policy_storage.latent_mean[:-1]),
                                                              torch.cat(policy_storage.latent_logvar[:-1]))
            self.task_inference_latent_rms.update(task_inference_latent.detach().clone())
        if self.pass_rim_level1_output_to_policy and self.norm_rim_level1_output:
            self.rim_level1_output_rms.update(torch.cat(policy_storage.brim_output_level1[:-1]).detach().clone())
        if self.pass_belief_to_policy and self.norm_belief:
            self.belief_rms.update(policy_storage.beliefs[:-1])
        if self.pass_task_to_policy and self.norm_task:
            self.task_rms.update(policy_storage.tasks[:-1])

    def evaluate_actions(self, embedded_state, latent, brim_output_level1, belief, task, action, return_action_mean=False):

        value, actor_features = self.forward(embedded_state, latent, brim_output_level1, belief, task)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        if not return_action_mean:
            return value, action_log_probs, dist_entropy
        else:
            return value, action_log_probs, dist_entropy, dist.mode(), dist.stddev


FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


def init(module, weight_init, bias_init, gain=1.0):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


# https://github.com/openai/baselines/blob/master/baselines/common/tf_util.py#L87
def init_normc_(weight, gain=1):
    weight.normal_(0, 1)
    weight *= gain / torch.sqrt(weight.pow(2).sum(1, keepdim=True))


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, init_std, min_std,
                 action_low, action_high, norm_actions_of_policy):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = nn.Parameter(np.log(torch.zeros(num_outputs) + init_std))
        self.min_std = torch.tensor([min_std]).to(device)

        # whether or not to conform to the action space given by the env
        # (scale / squash actions that the network outpus)
        self.norm_actions_of_policy = norm_actions_of_policy
        if len(np.unique(action_low)) == 1 and len(np.unique(action_high)) == 1:
            self.unique_action_limits = True
        else:
            self.unique_action_limits = False

        self.action_low = torch.from_numpy(action_low).to(device)
        self.action_high = torch.from_numpy(action_high).to(device)

    def forward(self, x):
        action_mean = self.fc_mean(x)
        if self.norm_actions_of_policy:
            if self.unique_action_limits and \
                    torch.unique(self.action_low) == -1 and \
                    torch.unique(self.action_high) == 1:
                action_mean = torch.tanh(action_mean)
            else:
                action_mean = torch.sigmoid(action_mean) * (self.action_high - self.action_low) + self.action_low
        std = torch.max(self.min_std, self.logstd.exp())
        return FixedNormal(action_mean, std)


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().reshape(1, -1)
        else:
            bias = self._bias.t().reshape(1, -1, 1, 1)

        return x + bias
