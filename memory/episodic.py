import torch
import torch.nn.functional as F
import torch.nn as nn
from memory.helpers import spatial_softmax, apply_alpha
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DND(nn.Module):
    def __init__(self, num_head, key_size, value_size, key_encoder_layer, value_encoder_layer, episode_len, policy_num_steps):
        super(DND, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head
        self.episode_len = episode_len
        self.exploration_batch_size = self.exploitation_batch_size = self.exploration_keys = \
            self.exploitation_keys = self.exploration_vals = self.exploitation_vals = \
            self.exploration_referenced_times = self.exploitation_referenced_times = self.exploration_RPE_read_modulation = self.exploitation_RPE_read_modulation =\
            self.exploration_step = self.exploitation_step = None

        # they should be shallow
        self.key_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(len(key_encoder_layer)):
            self.key_encoder.append(nn.Linear(curr_dim, key_encoder_layer[i]))
            # self.key_encoder.append(nn.ReLU())
            curr_dim = key_encoder_layer[i]
        self.key_encoder.append(nn.Linear(curr_dim, key_size))
        self.key_encoder = nn.Sequential(*self.key_encoder)

        self.value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(value_encoder_layer)):
            self.value_encoder.append(nn.Linear(curr_dim, value_encoder_layer[i]))
            # self.value_encoder.append(nn.ReLU())
            curr_dim = value_encoder_layer[i]
        self.value_encoder.append(nn.Linear(curr_dim, value_size))
        self.value_encoder = nn.Sequential(*self.value_encoder)

        self.query_encoder = nn.Linear(key_size, num_head*key_size)
        self.value_aggregator = nn.Linear(num_head*value_size, value_size)

    def prior(self, batch_size, activated_branch):
        if activated_branch == 'exploration':
            self.exploration_batch_size = batch_size
            self.exploration_keys = torch.zeros(size=(self.episode_len, batch_size, self.key_size), requires_grad=False, device=device)
            self.exploration_vals = torch.zeros(size=(self.episode_len, batch_size, self.value_size), requires_grad=False, device=device)
            self.exploration_referenced_times = torch.zeros(size=(self.episode_len, batch_size, 1), requires_grad=False, device=device)
            self.exploration_RPE_read_modulation = 1.0 * torch.ones(size=(self.episode_len, batch_size, 1), requires_grad=False, device=device)
            self.exploration_step = torch.zeros(size=(batch_size, 1), dtype=torch.long, requires_grad=False, device=device)
        elif activated_branch == 'exploitation':
            self.exploitation_batch_size = batch_size
            self.exploitation_keys = torch.zeros(size=(self.episode_len, batch_size, self.key_size), requires_grad=False, device=device)
            self.exploitation_vals = torch.zeros(size=(self.episode_len, batch_size, self.value_size), requires_grad=False, device=device)
            self.exploitation_referenced_times = torch.zeros(size=(self.episode_len, batch_size, 1), requires_grad=False, device=device)
            self.exploitation_RPE_read_modulation = 1.0 * torch.ones(size=(self.episode_len, batch_size, 1), requires_grad=False, device=device)
            self.exploitation_step = torch.zeros(size=(batch_size, 1), dtype=torch.long, requires_grad=False, device=device)
        else:
            raise NotImplementedError

    def reset(self, done_task, done_process_mdp, activated_branch):
        done_episode_idx = done_process_mdp.view(len(done_process_mdp)).nonzero(as_tuple=True)[0]
        if activated_branch == 'exploration':
            self.exploration_step[done_episode_idx] *= 0
        elif activated_branch == 'exploitation':
            self.exploitation_step[done_episode_idx] *= 0
        else:
            raise NotImplementedError

    def write(self, memory_key, memory_val, rpe, activated_branch):
        key = self.key_encoder(memory_key)
        value = self.value_encoder(memory_val)
        if activated_branch == 'exploration':
            self.exploration_keys[self.exploration_step, torch.arange(self.exploration_batch_size), :] = key
            self.exploration_vals[self.exploration_step, torch.arange(self.exploration_batch_size), :] = value
            if isinstance(rpe, torch.Tensor):
                self.exploration_RPE_read_modulation[self.exploration_step, torch.arange(self.exploration_batch_size), :] = torch.abs(rpe.detach())
            self.exploration_step = (self.exploration_step + 1) % self.episode_len
        elif activated_branch == 'exploitation':
            self.exploitation_keys[self.exploitation_step, torch.arange(self.exploitation_batch_size), :] = key
            self.exploitation_vals[self.exploitation_step, torch.arange(self.exploitation_batch_size), :] = value
            if isinstance(rpe, torch.Tensor):
                self.exploitation_RPE_read_modulation[self.exploitation_step, torch.arange(self.exploitation_batch_size), :] = torch.abs(rpe.detach())
            self.exploitation_step = (self.exploitation_step + 1) % self.episode_len
        else:
            raise NotImplementedError

    def read(self, query, activated_branch):
        if activated_branch == 'exploration':
            assert query.shape[0] == self.exploration_batch_size
            if len(self.exploration_step.nonzero(as_tuple=True)[0]) < self.exploration_batch_size:
                return torch.zeros((self.exploration_batch_size, self.value_size), device=device)
            keys = self.exploration_keys
            vals = self.exploration_vals
            RPE_read_modulation = self.exploration_RPE_read_modulation
            step = self.exploration_step
            batch_size = self.exploration_batch_size
            referenced_times = self.exploration_referenced_times
        elif activated_branch == 'exploitation':
            assert query.shape[0] == self.exploitation_batch_size
            if len(self.exploitation_step.nonzero(as_tuple=True)[0]) < self.exploitation_batch_size:
                return torch.zeros((self.exploitation_batch_size, self.value_size), device=device)
            keys = self.exploitation_keys
            vals = self.exploitation_vals
            RPE_read_modulation = self.exploitation_RPE_read_modulation
            step = self.exploitation_step
            batch_size = self.exploitation_batch_size
            referenced_times = self.exploitation_referenced_times
        else:
            raise NotImplementedError

        query = self.query_encoder(query).reshape(-1, self.num_head, self.key_size)
        results = []
        for i in range(batch_size):
            key_ = keys[:step[i], i, :].clone() * RPE_read_modulation[:step[i], i, :]
            query_ = query[i]
            # 1 = b, n = 7 , m = 48 . b = 1, m = 48, p = 4 - > b * n * p
            query_ = query_.permute(1, 0).unsqueeze(0)
            key_ = key_.unsqueeze(0)
            score = torch.bmm(key_, query_) / math.sqrt(self.key_size)
            prob = F.softmax(score, dim=1)
            tmp = prob.mean(-1).squeeze(0).unsqueeze(-1)
            referenced_times[:step[i], i, :] += tmp
            # b=1, n=4, m=7 . b=1 m=7 p=48 -> b*n*p
            prob = prob.permute(0, 2, 1)
            vals_ = vals[:step[i], i, :].clone().unsqueeze(0)
            result = torch.bmm(prob, vals_)
            result = self.value_aggregator(result.reshape(-1, self.num_head * self.value_size))
            results.append(result)
        if activated_branch == 'exploration':
            self.exploration_referenced_times = referenced_times
        if activated_branch == 'exploitation':
            self.exploitation_referenced_times = referenced_times
        results = torch.cat(results, dim=0)
        return results

    def get_done_process(self, done_process_mdp, activated_branch):
        if activated_branch == 'exploration':
            referenced_times = self.exploration_referenced_times
            keys = self.exploration_keys
            vals = self.exploration_vals
            RPE_read_modulation = self.exploration_RPE_read_modulation
        elif activated_branch == 'exploitation':
            referenced_times = self.exploitation_referenced_times
            keys = self.exploitation_keys
            vals = self.exploitation_vals
            RPE_read_modulation = self.exploitation_RPE_read_modulation
        else:
            raise NotImplementedError
        times = torch.range(1, self.episode_len).flip(dims=[-1]).unsqueeze(-1).unsqueeze(-1).expand(size=referenced_times.size()).to(device)
        referenced_times = referenced_times / times
        done_process_mdp = done_process_mdp.view(done_process_mdp.shape[0])
        idx = []
        idx_base = 0
        done_process_mdp_ = done_process_mdp.nonzero(as_tuple=True)[0]

        for i in range(len(done_process_mdp_)):
            _, idx_tmp = torch.topk(referenced_times[:, done_process_mdp_[i].item(), :].view(-1), dim=-1, k=(self.episode_len // 2))
            idx.append(idx_tmp + idx_base)
            idx_base += self.episode_len

        idx = torch.cat(idx, dim=-1)
        tmp_keys = keys[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_keys = tmp_keys.view(-1, self.key_size)[idx].view(len(done_process_mdp_), self.episode_len//2, -1)
        tmp_values = vals[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_values = tmp_values.view(-1, self.value_size)[idx].view(len(done_process_mdp_), self.episode_len//2, -1)
        tmp_RPE = RPE_read_modulation[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_RPE = tmp_RPE.view(-1, 1)[idx].view(len(done_process_mdp_), self.episode_len//2, -1)

        return ret_keys.detach(), ret_values.detach(), ret_RPE.detach()

    def compute_intrinsic_reward(self, memory_key):
        if len(self.exploration_step.nonzero(as_tuple=True)[0]) < self.exploration_batch_size:
            return torch.zeros((self.exploration_batch_size, 1), device=device)
        query = self.key_encoder(memory_key)
        keys = self.exploration_keys
        step = self.exploration_step
        batch_size = self.exploration_batch_size
        results = []
        for i in range(batch_size):
            key_ = keys[:step[i], i, :].clone()
            intrinsic_reward = torch.min(torch.sqrt(torch.sum((query[i] - key_)**2, dim=-1)))
            results.append(intrinsic_reward)
        results = torch.stack(results).unsqueeze(-1)
        return results