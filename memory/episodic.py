import torch
import torch.nn.functional as F
import torch.nn as nn
from memory.helpers import spatial_softmax, apply_alpha
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DND(nn.Module):
    def __init__(self, num_head, key_size, value_size,
                 key_encoder_layer, value_encoder_layer, episode_len,
                 general_key_encoder_layer,
                 general_value_encoder_layer,
                 general_query_encoder_layer,
                 read_memory_to_key_layer,
                 read_memory_to_value_layer,
                 rim_query_size,
                 rim_level1_hidden_size,
                 memory_state_embedding,
                 state_dim,
                 use_hebb
                 ):
        super(DND, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head
        self.episode_len = episode_len
        self.exploration_batch_size = self.exploitation_batch_size = self.exploration_keys = \
            self.exploitation_keys = self.exploration_vals = self.exploitation_vals = \
            self.exploration_referenced_times = self.exploitation_referenced_times = self.exploration_RPE_read_modulation = self.exploitation_RPE_read_modulation =\
            self.exploration_step = self.exploitation_step = None
        self.saved_keys = self.saved_values = None
        self.rim_level1_hidden_size = rim_level1_hidden_size
        self.memory_state_embedding = memory_state_embedding
        self.state_dim = state_dim
        self.use_hebb = use_hebb

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

        self.concat_key_encoder, self.concat_value_encoder, self.concat_query_encoder = self.initialise_encoder(
            key_size,
            value_size,
            general_key_encoder_layer,
            general_value_encoder_layer,
            general_query_encoder_layer)

        self.read_memory_to_value, self.read_memory_to_key = self.initialise_readout_attention(
            read_memory_to_key_layer,
            read_memory_to_value_layer,
            rim_query_size,
            value_size)

        self.state_encoder = nn.Linear(state_dim, memory_state_embedding)

    @staticmethod
    def initialise_encoder(key_size,
                           value_size,
                           general_key_encoder_layer,
                           general_value_encoder_layer,
                           general_query_encoder_layer
                           ):

        curr_dim = key_size
        key_encoder = nn.ModuleList([])
        for i in range(len(general_key_encoder_layer)):
            key_encoder.append(nn.Linear(curr_dim, general_key_encoder_layer[i]))
            # key_encoder.append(nn.ReLU())
            curr_dim = general_key_encoder_layer[i]
        key_encoder.append(nn.Linear(curr_dim, key_size))
        key_encoder = nn.Sequential(*key_encoder)

        value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(general_value_encoder_layer)):
            value_encoder.append(nn.Linear(curr_dim, general_value_encoder_layer[i]))
            # value_encoder.append(nn.ReLU())
            curr_dim = general_value_encoder_layer[i]
        value_encoder.append(nn.Linear(curr_dim, value_size))
        value_encoder = nn.Sequential(*value_encoder)

        query_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(len(general_query_encoder_layer)):
            query_encoder.append(nn.Linear(curr_dim, general_query_encoder_layer[i]))
            # query_encoder.append(nn.ReLU())
            curr_dim = general_query_encoder_layer[i]
        query_encoder.append(nn.Linear(curr_dim, key_size))
        query_encoder = nn.Sequential(*query_encoder)
        return key_encoder, value_encoder, query_encoder

    @staticmethod
    def initialise_readout_attention(read_memory_to_key_layer,
                                     read_memory_to_value_layer,
                                     rim_query_size,
                                     value_size,
                                     ):

        read_memory_to_value = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(read_memory_to_value_layer)):
            read_memory_to_value.append(nn.Linear(curr_dim, read_memory_to_value_layer[i]))
            # read_memory_to_value.append(nn.ReLU())
            curr_dim = read_memory_to_value_layer[i]
        read_memory_to_value.append(nn.Linear(curr_dim, value_size))
        read_memory_to_value = nn.Sequential(*read_memory_to_value)

        read_memory_to_key = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(read_memory_to_key_layer)):
            read_memory_to_key.append(nn.Linear(curr_dim, read_memory_to_key_layer[i]))
            # read_memory_to_key.append(nn.ReLU())
            curr_dim = read_memory_to_key_layer[i]
        read_memory_to_key.append(nn.Linear(curr_dim, rim_query_size))
        read_memory_to_key = nn.Sequential(*read_memory_to_key)

        return read_memory_to_value, read_memory_to_key,

    def prior(self, batch_size, activated_branch):
        if activated_branch == 'exploration':
            self.exploration_batch_size = batch_size
            self.exploration_keys = torch.zeros(size=(self.episode_len, batch_size, self.key_size), requires_grad=False, device=device)
            self.exploration_vals = torch.zeros(size=(self.episode_len, batch_size, self.value_size), requires_grad=False, device=device)
            self.exploration_referenced_times = torch.zeros(size=(self.episode_len, batch_size, 1), requires_grad=False, device=device)
            self.exploration_RPE_read_modulation = 1.0 * torch.ones(size=(self.episode_len, batch_size, 1), requires_grad=False, device=device)
            self.exploration_step = torch.zeros(size=(batch_size, 1), dtype=torch.long, requires_grad=False, device=device)
            if self.use_hebb:
                self.saved_keys = torch.zeros(size=(self.episode_len, batch_size, self.key_size - self.memory_state_embedding + self.state_dim), requires_grad=False, device=device)
                self.saved_values = torch.zeros(size=(self.episode_len, batch_size, self.value_size), requires_grad=False, device=device)
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

    def write(self, state, task_inference_latent, value, rpe, activated_branch):
        if self.use_hebb:
            self.saved_keys[self.exploration_step, torch.arange(self.exploration_batch_size), :] = torch.cat((state, task_inference_latent), dim=-1)
            self.saved_values[self.exploration_step, torch.arange(self.exploration_batch_size), :] = value
        state = self.state_encoder(state)
        key = self.concat_key_encoder(torch.cat((state, task_inference_latent), dim=-1)) # from memory.py
        key = self.key_encoder(key)
        value = self.concat_value_encoder(value) # from memory.py
        value = self.value_encoder(value)#.squeeze(0)
        if activated_branch == 'exploration':
            self.exploration_keys[self.exploration_step, torch.arange(self.exploration_batch_size), :] = key
            self.exploration_vals[self.exploration_step, torch.arange(self.exploration_batch_size), :] = value
            if isinstance(rpe, torch.Tensor):
                self.exploration_RPE_read_modulation[self.exploration_step, torch.arange(self.exploration_batch_size), :] = torch.abs(rpe.detach())
            self.exploration_step = (self.exploration_step + 1) % self.episode_len
        else:
            raise NotImplementedError

    def read(self, state, task_inference_latent, activated_branch):
        state = self.state_encoder(state)
        query = self.concat_query_encoder(torch.cat((state, task_inference_latent), dim=-1)) # from memory.py
        if activated_branch == 'exploration':
            assert query.shape[0] == self.exploration_batch_size
            if len(self.exploration_step.nonzero(as_tuple=True)[0]) < self.exploration_batch_size:
                results = torch.zeros((self.exploration_batch_size, self.value_size), device=device).unsqueeze(1)
                return self.read_memory_to_key(results), self.read_memory_to_value(results)
            keys = self.exploration_keys
            vals = self.exploration_vals
            RPE_read_modulation = self.exploration_RPE_read_modulation
            step = self.exploration_step
            batch_size = self.exploration_batch_size
            referenced_times = self.exploration_referenced_times
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
        results = torch.cat(results, dim=0).unsqueeze(1)
        k = self.read_memory_to_key(results)
        v = self.read_memory_to_value(results)
        return k, v

    def get_done_process(self, done_process_mdp, activated_branch):
        if activated_branch == 'exploration':
            referenced_times = self.exploration_referenced_times
            keys = self.saved_keys
            vals = self.saved_values
            RPE_read_modulation = self.exploration_RPE_read_modulation
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
        ret_keys = tmp_keys.view(-1, self.key_size - self.memory_state_embedding + self.state_dim)[idx].view(len(done_process_mdp_), self.episode_len//2, -1)
        tmp_values = vals[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_values = tmp_values.view(-1, self.value_size)[idx].view(len(done_process_mdp_), self.episode_len//2, -1)
        tmp_RPE = RPE_read_modulation[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_RPE = tmp_RPE.view(-1, 1)[idx].view(len(done_process_mdp_), self.episode_len//2, -1)

        ret_state = ret_keys[:, :, :self.state_dim].detach()
        ret_task_inf_latent = ret_keys[:, :, self.state_dim:].detach()

        return ret_state, ret_task_inf_latent, ret_values.detach(), ret_RPE.detach()

    def compute_intrinsic_reward(self, state, task_inf_latent):
        state = self.state_encoder(state)
        memory_key = self.concat_key_encoder(torch.cat((state, task_inf_latent), dim=-1))
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