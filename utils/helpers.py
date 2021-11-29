import os
import gym
import pickle
import random
import warnings
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from environments.parallel_envs import make_vec_envs
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_env(env, args, indices=None, state=None):
    """ env can be many environments or just one """
    # reset all environments
    if indices is not None:
        assert not isinstance(indices[0], bool)
    if (indices is None) or (len(indices) == args.num_processes):
        state = env.reset().to(device)
    # reset only the ones given by indices
    else:
        assert state is not None
        for i in indices:
            state[i] = env.reset(index=i)

    belief = torch.from_numpy(env.get_belief()).to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).to(device) if args.pass_task_to_policy else None
        
    return state, belief, task


def env_step(env, action, args):

    next_obs, reward, done, infos = env.step(action.detach())

    if isinstance(next_obs, list):
        next_obs = [o.to(device) for o in next_obs]
    else:
        next_obs = next_obs.to(device)
    if isinstance(reward, list):
        reward = [r.to(device) for r in reward]
    else:
        reward = reward.to(device)

    belief = torch.from_numpy(env.get_belief()).float().to(device) if args.pass_belief_to_policy else None
    task = torch.from_numpy(env.get_task()).to(device) if args.pass_task_to_policy else None

    return [next_obs, belief, task], reward, done, infos


def select_action(args,
                  policy,
                  deterministic,
                  belief=None,
                  task=None,
                  latent_sample=None, latent_mean=None, latent_logvar=None,
                  brim_output_level1=None,
                  policy_embedded_state=None):
    """ Select action using the policy. """
    latent = get_latent_for_policy(sample_embeddings=args.sample_embeddings, add_nonlinearity_to_latent=args.add_nonlinearity_to_latent, latent_sample=latent_sample, latent_mean=latent_mean, latent_logvar=latent_logvar)
    action = policy.act(embedded_state=policy_embedded_state, latent=latent, brim_output_level1=brim_output_level1, belief=belief, task=task, deterministic=deterministic)
    if isinstance(action, list) or isinstance(action, tuple):
        value, action, action_log_prob = action
    else:
        value = None
        action_log_prob = None
    action = action.to(device)
    return value, action, action_log_prob


def get_latent_for_policy(sample_embeddings, add_nonlinearity_to_latent, latent_sample=None, latent_mean=None, latent_logvar=None):

    if (latent_sample is None) and (latent_mean is None) and (latent_logvar is None):
        return None

    if add_nonlinearity_to_latent:
        latent_sample = F.relu(latent_sample)
        latent_mean = F.relu(latent_mean)
        latent_logvar = F.relu(latent_logvar)

    if sample_embeddings:
        latent = latent_sample
    else:
        latent = torch.cat((latent_mean, latent_logvar), dim=-1)

    if latent.shape[0] == 1 and latent.dim() == 3:
        latent = latent.squeeze(0)

    return latent


def update_encoding(brim_core, policy, next_obs, action, reward, done, task_inference_hidden_state, brim_hidden_state, activated_branch, done_episode, rpe):
    # reset hidden state of the recurrent net when we reset the task
    if done is not None:
        task_inference_hidden_state, brim_hidden_state = brim_core.reset_hidden(policy, task_inference_hidden_state, brim_hidden_state, done_task=done, done_episode=done_episode, activated_branch=activated_branch)

    with torch.no_grad():
        if activated_branch == 'exploration':
            brim_output1, brim_output3, brim_output5, brim_hidden_state,\
            latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, exploration_policy_embedded_state = brim_core.forward_exploration_branch(
                actions=action.float(),
                states=next_obs,
                rewards=reward,
                task_inference_hidden_state=task_inference_hidden_state,
                brim_hidden_state=brim_hidden_state,
                sample=True,
                return_prior=False,
                detach_every=None,
                policy=policy,
                rpe=rpe)
            return brim_output1, brim_output3, brim_output5, brim_hidden_state, latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, exploration_policy_embedded_state
        if activated_branch == 'exploitation':
            brim_output2, brim_output4, brim_output5, brim_hidden_state, \
            latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, exploitation_policy_embedded_state = brim_core.forward_exploitation_branch(
                actions=action.float(),
                states=next_obs,
                rewards=reward,
                task_inference_hidden_state=task_inference_hidden_state,
                brim_hidden_state=brim_hidden_state,
                sample=True,
                return_prior=False,
                detach_every=None,
                policy=policy,
                rpe=rpe)
            return brim_output2, brim_output4, brim_output5, brim_hidden_state, latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, exploitation_policy_embedded_state


def compute_intrinsic_reward(rew_raw,
                             rew_normalised,
                             latent,
                             prev_state,
                             next_state,
                             action,
                             state_decoder,
                             action_decoder,
                             reward_decoder,
                             decode_action,
                             decode_reward,
                             state_prediction_running_normalizer,
                             action_prediction_running_normalizer,
                             reward_prediction_running_normalizer,
                             state_prediction_intrinsic_reward_coef,
                             action_prediction_intrinsic_reward_coef,
                             reward_prediction_intrinsic_reward_coef,
                             rew_pred_type,
                             extrinsic_reward_intrinsic_reward_coef,
                             itr_idx,
                             num_updates,
                             memory,
                             episodic_reward,
                             episodic_reward_coef,
                             task_inf_latent,
                             epi_reward_running_normalizer,
                             exponential_temp_epi,
                             intrinsic_reward_running_normalizer,
                             state_encoder):
    if decode_action:
        action_pred = action_decoder(latent_state=latent, state=prev_state, next_state=next_state, n_step_next_state=None, n_step_action_prediction=False)[0].detach()
        action_error = F.nll_loss(action_pred, action.squeeze(-1).long(), reduction='none').unsqueeze(-1)
    else:
        action_error = 0.0
    if decode_reward:
        reward_pred = reward_decoder(latent_state=latent, next_state=next_state, prev_state=prev_state, action=action, n_step_reward_prediction=False)[0].detach()
        if rew_pred_type == 'categorical':
            reward_pred = F.softmax(reward_pred, dim=-1)
        elif rew_pred_type == 'bernoulli':
            reward_pred = torch.sigmoid(reward_pred)

        rew_target = (rew_raw == 1).float()
        if rew_pred_type == 'deterministic':
            reward_error = (reward_pred - rew_raw).pow(2)
        elif rew_pred_type in ['categorical', 'bernoulli']:
            reward_error = F.binary_cross_entropy(reward_pred, rew_target, reduction='none')
        else:
            raise NotImplementedError
    else:
        reward_error = 0
    if memory is not None and episodic_reward:
        epi_reward = memory.compute_intrinsic_reward(next_state, task_inf_latent).detach()
    else:
        epi_reward = 0.0
    annealing_tmp = 1 - (itr_idx / num_updates)
    state_pred = state_decoder(latent_state=latent, state=prev_state, action=action, n_step_action=None,
                               n_step_state_prediction=False)[0].detach()
    state_error = (state_pred - state_encoder(next_state).detach()).pow(2).mean(dim=-1).unsqueeze(-1)
    intrinsic_reward = state_error * state_prediction_intrinsic_reward_coef + \
    action_error * action_prediction_intrinsic_reward_coef + \
    reward_error * reward_prediction_intrinsic_reward_coef
    if episodic_reward:
        modulation = torch.minimum(torch.maximum(intrinsic_reward,
                                    torch.ones_like(state_error) * 1.0), torch.ones_like(state_error) * 1.0)
        intrinsic_rew_raw = (modulation * (episodic_reward_coef * epi_reward)) * annealing_tmp + \
            rew_raw * extrinsic_reward_intrinsic_reward_coef
    else:
        intrinsic_rew_raw = intrinsic_reward * annealing_tmp + rew_raw * extrinsic_reward_intrinsic_reward_coef

    intrinsic_rew_normalised = (intrinsic_rew_raw - intrinsic_reward_running_normalizer.mean) / torch.sqrt(intrinsic_reward_running_normalizer.var + 1e-8)

    # print('state_error ', state_error)
    # print('action_error ', action_error)
    # print('reward_error ', reward_error)
    # print('intrinsic_reward ', intrinsic_reward)
    # print('modulation ', modulation)
    # print('epi_reward ', epi_reward)
    # print('intrinsic_rew_raw ', intrinsic_rew_raw)

    if isinstance(state_error, torch.Tensor):
        state_error = state_error.detach()
    
    if isinstance(intrinsic_rew_raw, torch.Tensor):
        intrinsic_rew_raw = intrinsic_rew_raw.detach()
    
    if isinstance(intrinsic_rew_normalised, torch.Tensor):
        intrinsic_rew_normalised = intrinsic_rew_normalised.detach()
    
    if isinstance(action_error, torch.Tensor):
        action_error = action_error.detach()
    
    if isinstance(reward_error, torch.Tensor):
        reward_error = reward_error.detach()

    if isinstance(epi_reward, torch.Tensor):
        epi_reward = epi_reward.detach()

    return intrinsic_rew_raw, intrinsic_rew_normalised, state_error, action_error, reward_error, epi_reward


def count_params_number(base2final, policy):
    params_number = sum(p.numel() for p in base2final.brim_core.parameters() if p.requires_grad)* 4/5
    params_number += sum(p.numel() for p in base2final.reward_decoder.parameters() if p.requires_grad)
    params_number += sum(p.numel() for p in base2final.action_decoder.parameters() if p.requires_grad)
    params_number += sum(p.numel() for p in base2final.state_decoder.parameters() if p.requires_grad)
    params_number += sum(p.numel() for p in base2final.exploration_value_decoder.parameters() if p.requires_grad)
    params_number += sum(p.numel() for p in policy.parameters() if p.requires_grad)
    return params_number


def seed(seed, deterministic_execution=False):
    print('Seeding random, torch, numpy.')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    if deterministic_execution:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('Note that due to parallel processing results will be similar but not identical. '
              'If you want identical results, use --num_processes 1 and --deterministic_execution True '
              '(only recommended for debugging).')


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def recompute_embeddings(
        policy,
        policy_storage,
        brim_core,
        sample,
        update_idx,
        detach_every,
        activated_branch,
):
    # get the prior
    latent_sample = [policy_storage.latent_samples[0].detach().clone()]
    latent_mean = [policy_storage.latent_mean[0].detach().clone()]
    latent_logvar = [policy_storage.latent_logvar[0].detach().clone()]

    latent_sample[0].requires_grad = True
    latent_mean[0].requires_grad = True
    latent_logvar[0].requires_grad = True

    brim_output_level1 = [policy_storage.brim_output_level1[0].detach().clone()]
    brim_output_level2 = [policy_storage.brim_output_level2[0].detach().clone()]
    brim_output_level3 = [policy_storage.brim_output_level3[0].detach().clone()]
    policy_embedded_state = [policy_storage.policy_embedded_state[0].detach().clone()]

    brim_output_level1[0].requires_grad = True
    brim_output_level2[0].requires_grad = True
    brim_output_level3[0].requires_grad = True
    policy_embedded_state[0].requires_grad = True

    # loop through experience and update hidden state
    # (we need to loop because we sometimes need to reset the hidden state)
    task_inference_hidden_state = policy_storage.task_inference_hidden_states[0].detach()
    brim_hidden_state = policy_storage.brim_hidden_states[0].detach()
    policy.state_encoder.prior(policy_storage.actions.shape[1])
    policy.state_encoder.detach_hidden_state()
    memory = brim_core.brim.model.memory
    if memory is not None:
        memory.prior(policy_storage.actions.shape[1], activated_branch)
    for i in range(policy_storage.actions.shape[0]):
        # reset hidden state of the GRU when we reset the task
        task_inference_hidden_state, brim_hidden_state = brim_core.reset_hidden(policy, task_inference_hidden_state, brim_hidden_state, policy_storage.done[i + 1], done_episode=policy_storage.done_episode[i + 1], activated_branch=activated_branch)

        if activated_branch == 'exploration':
            brim_output1, brim_output3, brim_output5, brim_hidden_state, \
            latent_sample_, latent_mean_, latent_logvar_, task_inference_hidden_state, exploration_policy_embedded_state = brim_core.forward_exploration_branch(
                actions=policy_storage.actions.float()[i:i + 1],
                states=policy_storage.next_state[i:i + 1],
                rewards=policy_storage.rewards_raw[i:i + 1],
                task_inference_hidden_state=task_inference_hidden_state,
                brim_hidden_state=brim_hidden_state,
                sample=sample,
                return_prior=False,
                detach_every=detach_every,
                policy=policy)
            latent_sample.append(latent_sample_)
            latent_mean.append(latent_mean_)
            latent_logvar.append(latent_logvar_)

            brim_output_level1.append(brim_output1)
            brim_output_level2.append(brim_output3)
            brim_output_level3.append(brim_output5)
            policy_embedded_state.append(exploration_policy_embedded_state)

        if activated_branch == 'exploitation':
            brim_output2, brim_output4, brim_output5, brim_hidden_state, \
            latent_sample_, latent_mean_, latent_logvar_, task_inference_hidden_state, exploitation_policy_embedded_state = brim_core.forward_exploitation_branch(
                actions=policy_storage.actions.float()[i:i + 1],
                states=policy_storage.next_state[i:i + 1],
                rewards=policy_storage.rewards_raw[i:i + 1],
                task_inference_hidden_state=task_inference_hidden_state,
                brim_hidden_state=brim_hidden_state,
                sample=sample,
                return_prior=False,
                detach_every=detach_every,
                policy=policy)
            latent_sample.append(latent_sample_)
            latent_mean.append(latent_mean_)
            latent_logvar.append(latent_logvar_)

            brim_output_level1.append(brim_output2)
            brim_output_level2.append(brim_output4)
            brim_output_level3.append(brim_output5)
            policy_embedded_state.append(exploitation_policy_embedded_state)

    if update_idx == 0 and activated_branch == 'exploration':
        try:
            assert (torch.cat(policy_storage.latent_mean) - torch.cat(latent_mean)).sum() == 0
            assert (torch.cat(policy_storage.latent_logvar) - torch.cat(latent_logvar)).sum() == 0
        except AssertionError:
            warnings.warn('You are not recomputing the embeddings correctly!')
            import pdb
            pdb.set_trace()

    policy_storage.latent_samples = latent_sample
    policy_storage.latent_mean = latent_mean
    policy_storage.latent_logvar = latent_logvar

    policy_storage.brim_output_level1 = brim_output_level1
    policy_storage.brim_output_level2 = brim_output_level2
    policy_storage.brim_output_level3 = brim_output_level3
    policy_storage.policy_embedded_state = policy_embedded_state


class FeatureExtractor(nn.Module):
    """ Used for extrating features for states/actions/rewards """

    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self.output_size = output_size
        self.activation_function = activation_function
        if self.output_size != 0:
            self.fc = nn.Linear(input_size, output_size)
        else:
            self.fc = None

    def forward(self, inputs):
        if self.output_size != 0:
            return self.activation_function(self.fc(inputs))
        else:
            return torch.zeros(0, ).to(device)


def sample_gaussian(mu, logvar, num=None):
    std = torch.exp(0.5 * logvar)
    if num is not None:
        std = std.repeat(num, 1)
        mu = mu.repeat(num, 1)
    eps = torch.randn_like(std)
    return mu + std * eps


def save_obj(obj, folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(folder, name):
    filename = os.path.join(folder, name + '.pkl')
    with open(filename, 'rb') as f:
        with torch.loading_context(map_location=device):
            return pickle.load(f)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float().to(device)
        self.var = torch.ones(shape).float().to(device)
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))


def get_task_dim(args):
    env = make_vec_envs(env_name=args.env_name, seed=args.seed, num_processes=args.num_processes,
                        gamma=args.policy_gamma, device=device,
                        episodes_per_task=args.max_rollouts_per_task,
                        normalise_rew=args.norm_rew_for_policy, ret_rms=None,
                        )
    return env.task_dim


def get_num_tasks(args):
    env = gym.make(args.env_name)
    try:
        num_tasks = env.num_tasks
    except AttributeError:
        num_tasks = None
    return num_tasks


def clip(value, low, high):
    """Imitates `{np,tf}.clip`.

    `torch.clamp` doesn't support tensor valued low/high so this provides the
    clip functionality.

    """
    low, high = torch.tensor(low), torch.tensor(high)

    assert torch.all(low <= high), (low, high)

    clipped_value = torch.max(torch.min(value, high), low)
    return clipped_value


def flat_obs(obs):
    return torch.flatten(obs).view(1, -1)

def image_obs(obs):
    return obs[:, 0:-1].view(-1, 7, 7, 3)


class SimpleVision(nn.Module):
    def __init__(self, state_embedding_size):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.fc1 = nn.Linear(3 * 3 * 8, 64)
        self.fc2 = nn.Linear(64, state_embedding_size-1)

    def detach_hidden_state(self):
        pass

    def prior(self, batch_size):
        pass

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
