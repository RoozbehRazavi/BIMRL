import torch.nn as nn
import torch
import torch.nn.functional as F
from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Hebbian(nn.Module):
    def __init__(self, num_head, w_max, key_size, value_size, key_encoder_layer, value_encoder_layer, hebb_learning_rate,
                 rim_query_size,
                 general_query_encoder_layer,
                 state_dim,
                 memory_state_embedding,
                 read_memory_to_key_layer,
                 read_memory_to_value_layer,
                 ):
        super(Hebbian, self).__init__()
        num_head = 1
        self.learning_rate = hebb_learning_rate
        self.key_size = key_size
        self.num_head = num_head
        self.value_size = value_size
        self.w_max = w_max
        self.rim_query_size = rim_query_size

        self.normalize_key = utl.RunningMeanStd(shape=(self.key_size))
        self.normalize_value = utl.RunningMeanStd(shape=(self.value_size))

        key_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(len(key_encoder_layer)):
            key_encoder.append(nn.Linear(curr_dim, key_encoder_layer[i]))
            curr_dim = key_encoder_layer[i]
        key_encoder.append(nn.Linear(curr_dim, key_size))
        key_encoder.append(nn.ReLU())
        self.key_encoder = nn.Sequential(*key_encoder)

        value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(value_encoder_layer)):
            value_encoder.append(nn.Linear(curr_dim, value_encoder_layer[i]))
            curr_dim = value_encoder_layer[i]
        value_encoder.append(nn.Linear(curr_dim, self.value_size))
        value_encoder.append(nn.ReLU())
        self.value_encoder = nn.Sequential(*value_encoder)

        self.query_encoder = nn.Linear(key_size, num_head * self.key_size)
        self.value_aggregator = nn.Linear(num_head * value_size, value_size)
        self.concat_query_encoder = self.initialise_encoder(key_size, general_query_encoder_layer)
        self.state_encoder = nn.Linear(state_dim, memory_state_embedding)

        self.read_memory_to_value, self.read_memory_to_key = self.initialise_readout_attention(
            read_memory_to_key_layer,
            read_memory_to_value_layer,
            rim_query_size,
            value_size)

        self.exploitation_w_assoc = self.exploitation_batch_size = self.exploration_w_assoc = self.exploration_batch_size = None
        self.exploration_write_flag = None

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

    @staticmethod
    def initialise_encoder(key_size,
                           general_query_encoder_layer
                           ):

        query_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(len(general_query_encoder_layer)):
            query_encoder.append(nn.Linear(curr_dim, general_query_encoder_layer[i]))
            # query_encoder.append(nn.ReLU())
            curr_dim = general_query_encoder_layer[i]
        query_encoder.append(nn.Linear(curr_dim, key_size))
        query_encoder = nn.Sequential(*query_encoder)
        return query_encoder

    def prior(self, batch_size, activated_branch):
        if activated_branch == 'exploration':
            self.exploration_batch_size = batch_size
            self.exploration_w_assoc = torch.zeros((self.exploration_batch_size, self.key_size, self.value_size), requires_grad=False, device=device)
            #torch.nn.init.normal_(self.exploration_w_assoc, mean=0, std=0.1)
            
            #self.main_init = self.exploration_w_assoc.clone()
            self.exploration_write_flag = torch.zeros(size=(batch_size, 1), dtype=torch.long, requires_grad=False, device=device)
        else:
            raise NotImplementedError

    def reset(self, done_task, activated_branch):
        done_task_idx = done_task.view(len(done_task)).nonzero(as_tuple=True)[0]
        tmp = torch.zeros(
            size=(len(done_task_idx), self.key_size, self.value_size),
            requires_grad=False, device=device)
        #torch.nn.init.normal_(tmp, mean=0, std=0.1)
        
        #tmp = self.main_init[done_task_idx]
        if activated_branch == 'exploration':
            self.exploration_w_assoc[done_task_idx] = tmp
            self.exploration_write_flag[done_task_idx] = torch.zeros(size=(len(done_task_idx), 1), device=device, requires_grad=False, dtype=torch.long)
        else:
            raise NotImplementedError

    def write(self, state, task_inference_latent, value, modulation, done_process_mdp, activated_branch, A, B): 
        state = self.state_encoder(state)
        key = self.key_encoder(torch.cat((state, task_inference_latent), dim=-1))
        value = self.value_encoder(value)
        done_process_mdp = done_process_mdp.view(-1).nonzero(as_tuple=True)[0]
        self.exploration_write_flag[done_process_mdp] = torch.ones(size=(len(done_process_mdp), 1), device=device, requires_grad=False, dtype=torch.long)
        batch_size = len(done_process_mdp)
        value = modulation * value

        self.normalize_key.update(key.detach().clone())
        self.normalize_value.update(value.detach().clone())
        key = (key - self.normalize_key.mean) / torch.sqrt(self.normalize_key.var + 1e-8)
        value = (value - self.normalize_value.mean) / torch.sqrt(self.normalize_value.var + 1e-8)

        correlation = torch.bmm(key.permute(0, 2, 1), value)
        regularization = torch.bmm(key.permute(0, 2, 1), key)
        if activated_branch == 'exploration':
            A = A.expand(batch_size, -1, -1)
            B = B.expand(batch_size, -1, -1)
            for i in range(4):
                a1 = torch.bmm(A, (self.w_max - self.exploration_w_assoc[done_process_mdp].clone()).permute(0, 2, 1))
                a2 = torch.bmm(a1, correlation)
                a3 = torch.bmm(B, self.exploration_w_assoc[done_process_mdp].clone().permute(0, 2, 1))
                a4 = torch.bmm(a3, regularization).permute(0, 2, 1)
                delta_w = a2 - a4
                self.exploration_w_assoc[done_process_mdp] = self.exploration_w_assoc[done_process_mdp].clone() + self.learning_rate * delta_w
        else:
            raise NotImplementedError

    def read(self, state, task_inference_latent, activated_branch):
        state = self.state_encoder(state)
        query = self.concat_query_encoder(torch.cat((state, task_inference_latent), dim=-1))  # from memory.py
        query = F.relu(self.query_encoder(query).reshape(-1, self.num_head, self.key_size))
        if activated_branch == 'exploration':
            w_assoc = self.exploration_w_assoc.clone()
            batch_size = self.exploration_batch_size
        elif activated_branch == 'exploitation':
            w_assoc = self.exploitation_w_assoc.clone()
            batch_size = self.exploitation_batch_size
        else:
            raise NotImplementedError
        query = (query - self.normalize_key.mean) / torch.sqrt(self.normalize_key.var + 1e-8)
        value = torch.bmm(query, w_assoc)
        value = torch.mul(value, self.normalize_value.var) + self.normalize_value.mean
        value = value.reshape(batch_size, self.num_head*self.value_size)
        value = self.value_aggregator(value)
        k = self.read_memory_to_key(value)
        v = self.read_memory_to_value(value)
        exploration_write_flag = (1 - self.exploration_write_flag).view(-1).nonzero(as_tuple=True)[0]
        k[exploration_write_flag] = torch.zeros(size=(len(exploration_write_flag), self.rim_query_size), device=device)
        v[exploration_write_flag] = torch.zeros(size=(len(exploration_write_flag), self.value_size), device=device)
        return k.unsqueeze(1), v.unsqueeze(1)

