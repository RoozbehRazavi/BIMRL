import torch
import torch.nn.functional as F
import torch.nn as nn
from memory.helpers import spatial_softmax, apply_alpha
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class GroupLinearLayer(nn.Module):

    def __init__(self, din, dout, num_blocks):
        super(GroupLinearLayer, self).__init__()

        self.w = nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(self, x):
        x = x.permute(1, 0, 2)

        x = torch.bmm(x, self.w)
        return x.permute(1, 0, 2)


class DND(nn.Module):
    def __init__(self, num_head, key_size, value_size, key_encoder_layer, value_encoder_layer, query_encoder_layer, policy_num_steps, max_rollouts_per_task):
        super(DND, self).__init__()
        self.key_size = key_size
        self.value_size =value_size
        self.num_head = num_head
        self.episode_len = policy_num_steps // max_rollouts_per_task
        self.kernel = 'l2'
        self.batch_size = self.keys = self.vals = self.referenced_times = self.RPE_read_modulation = self.step = None

        # they should be shallow
        self.key_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(key_encoder_layer):
            self.key_encoder.append(nn.Linear(curr_dim, key_encoder_layer[i]))
            self.key_encoder.append(nn.ReLU())
        self.key_encoder.append(nn.Linear(curr_dim, key_size))
        self.key_encoder = nn.Sequential(*self.key_encoder)

        self.value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(value_encoder_layer):
            self.value_encoder.append(nn.Linear(curr_dim, value_encoder_layer[i]))
            self.value_encoder.append(nn.ReLU())
        self.value_encoder.append(nn.Linear(curr_dim, value_size))
        self.value_encoder = nn.Sequential(*self.value_encoder)

        self.query_encoder = nn.Linear(key_size, num_head*key_size)
        self.value_aggregator = nn.Linear(num_head*value_size, value_size)

    def prior(self, batch_size):
        self.batch_size = batch_size
        self.keys = torch.zeros(size=(self.episode_len, batch_size, self.key_size), device=device)
        self.vals = torch.zeros(size=(self.episode_len, batch_size, self.key_size), device=device)
        self.referenced_times = torch.zeros(size=(self.episode_len, batch_size, 1), device=device)
        self.RPE_read_modulation = torch.zeros(size=(self.episode_len, batch_size, 1), device=device)
        self.step = torch.zeros(size=(self.batch_size, 1), dtype=torch.long, device=device)

    def reset(self, done_process_mdp):
        self.step[done_process_mdp.view(len(done_process_mdp)).nonzero(as_tuple=True)[0]] *= 0

    def write(self, memory_key, memory_val, rpe):
        key = self.key_encoder(memory_key)
        value = self.value_encoder(memory_val)
        self.keys[self.step, torch.arange(self.batch_size), :] = key
        self.vals[self.step, torch.arange(self.batch_size), :] = value
        self.RPE_read_modulation[self.step, torch.arange(self.batch_size), :] = torch.abs(rpe.detach())
        self.step = (self.step + 1) % self.episode_len

    def read(self, query):
        assert query.shape[0] == self.batch_size
        if len(self.step.nonzero(as_tuple=True)[0]) < self.batch_size:
            return torch.zeros((self.batch_size, self.memory_brim_hidden1_dim + self.memory_brim_hidden2_dim), device=device)
        query = self.query_encoder(query).reshape(-1, self.num_head, self.key_size)
        results = []
        for i in range(self.batch_size):
            key_ = self.keys[:self.step[i], i, :].clone() * self.RPE_read_modulation[:self.step[i], i, :]
            query_ = query[i]
            key_ = key_.permute(1, 0, 2)
            query_ = query_.permute(0, 2, 1)
            a = torch.bmm(key_, query_)
            weight = F.softmax(a, dim=1)
            result = self._get_memory(weight, self.step[i], i)
            result = self.value_aggregator(result.reshape(-1, self.num_head * self.value_size))
            results.append(result)
        results = torch.stack(results, dim=0)
        return results

    def _get_memory(self, weight, step, i):
        self.referenced_times[:step, i, :] += weight
        res = apply_alpha(weight.clone(), self.vals[:step, i, :].clone())
        return res

    def get_done_process(self, done_process_mdp):
        times = torch.range(1, self.episode_len).flip(dims=[-1]).unsqueeze(-1).unsqueeze(-1).expand(size=self.referenced_times.size()).to(device)
        self.referenced_times = self.referenced_times / times
        done_process_mdp = done_process_mdp.view(done_process_mdp.shape[0])
        idx = []
        idx_base = 0
        done_process_mdp_ = done_process_mdp.nonzero(as_tuple=True)[0]

        for i in range(sum(done_process_mdp)):
            _, idx_tmp = torch.topk(self.referenced_times[:, done_process_mdp_[i], :].view(-1), dim=-1, k=(self.episode_len // 2))
            idx.append(idx_tmp + idx_base)
            idx_base += self.episode_len

        idx = torch.cat(idx, dim=-1)
        tmp_keys = self.keys[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_keys = tmp_keys.view(-1, self.key_size)[idx].view(torch.sum(done_process_mdp), self.episode_len//2, -1)
        tmp_values = self.vals[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_values = tmp_values.view(-1, self.value_size)[idx].view(torch.sum(done_process_mdp), self.episode_len//2, -1)
        tmp_RPE = self.RPE_read_modulation[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_RPE = tmp_RPE.view(-1, 1)[idx].view(torch.sum(done_process_mdp), self.episode_len//2, -1)

        return ret_keys.detach(), ret_values.detach(), ret_RPE.detach()

