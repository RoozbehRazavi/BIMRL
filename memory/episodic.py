import torch
import torch.nn.functional as F
import torch.nn as nn
from memory.helpers import spatial_softmax, apply_alpha
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DND(nn.Module):
    def __init__(self, args, num_head):
        super(DND, self).__init__()
        self.args = args
        self.num_head = num_head
        self.memory_brim_hidden1_dim = self.memory_brim_hidden2_dim = self.args.brim_hidden_size[0]
        self.episode_len = args.policy_num_steps // args.max_rollouts_per_task
        self.kernel = 'l2'
        self.batch_size = self.keys = self.vals = self.referenced_times = self.RPE_read_modulation = self.step = None
        self.linear_output = torch.nn.Linear(self.num_head*2*self.args.brim_hidden_size[0], 2*self.args.brim_hidden_size[0])

    def prior(self, batch_size):
        self.batch_size = batch_size
        self.keys = torch.zeros(size=(self.episode_len, batch_size, self.args.state_embedding_size), device=device)
        self.vals = torch.zeros(size=(self.episode_len, batch_size, self.memory_brim_hidden1_dim + self.memory_brim_hidden2_dim), device=device)
        self.referenced_times = torch.zeros(size=(self.episode_len, batch_size, 1), device=device)
        self.RPE_read_modulation = torch.zeros(size=(self.episode_len, batch_size, 1), device=device)
        self.step = torch.zeros(size=(self.batch_size, 1), dtype=torch.long, device=device)

    def reset(self, done_process_mdp):
        self.step[done_process_mdp.view(len(done_process_mdp)).nonzero(as_tuple=True)[0]] *= 0

    def write(self, memory_key, memory_val, RPE):
        self.keys[self.step] = memory_key
        self.vals[self.step] = memory_val
        self.RPE_read_modulation[self.step] = torch.abs(RPE.detach())
        self.step = (self.step + 1) % self.episode_len

    def read(self, query):
        if len(self.step.nonzero(as_tuple=True)[0]) < self.batch_size:
            return torch.zeros((self.batch_size, self.memory_brim_hidden1_dim + self.memory_brim_hidden2_dim), device=device)

        min_step = self.step.view(len(self.step)).min()
        K = self.keys[:min_step, :, :].clone() * self.RPE_read_modulation[:min_step, :, :]
        Q = query
        K = K.permute(1, 0, 2)
        Q = Q.permute(0, 2, 1)
        A = torch.bmm(K, Q)
        weight = F.softmax(A, dim=1)
        result = self._get_memory(weight, min_step)
        result = result.reshape(self.batch_size, self.num_head*2*self.args.brim_hidden_size[0])
        result = self.linear_output(result)
        return result

    def _get_memory(self, weight, min_step):
        ref_weight = weight.sum(dim=-1)
        ref_weight = ref_weight.permute(1, 0).unsqueeze(-1)
        self.referenced_times[:min_step] += ref_weight
        res = apply_alpha(weight.clone(), self.vals[:min_step].clone())
        return res

    def get_done_process(self, done_process_mdp):
        # use self.referenced_times for get best memories
        # TODO normalized history #TODO fix this shit
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
        ret_keys = tmp_keys.view(-1, self.args.state_embedding_size)[idx].view(torch.sum(done_process_mdp), self.episode_len//2, -1)
        tmp_values = self.vals[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_values = tmp_values.view(-1, 2*self.args.brim_hidden_size[0])[idx].view(torch.sum(done_process_mdp), self.episode_len//2, -1)
        tmp_RPE = self.RPE_read_modulation[:, done_process_mdp.nonzero(as_tuple=True)[0], :]
        ret_RPE = tmp_RPE.view(-1, 1)[idx].view(torch.sum(done_process_mdp), self.episode_len//2, -1)

        return ret_keys.detach(), ret_values.detach(), ret_RPE.detach()

