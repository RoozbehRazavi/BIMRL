import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Hebbian(nn.Module):
    def __init__(self, num_head, w_max, key_size, value_size, key_encoder_layer, value_encoder_layer):
        super(Hebbian, self).__init__()
        self.saved_keys_num = 100
        self.key_size = key_size
        self.num_head = num_head
        self.value_size = value_size
        self.w_max = w_max
        # hebbian parameters
        self.A = nn.Linear(key_size, value_size, bias=False)
        self.B = nn.Linear(value_size, value_size, bias=False)

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

        self.query_encoder = nn.Linear(key_size, num_head * key_size)
        self.value_aggregator = nn.Linear(num_head * value_size, value_size)

        self.saved_keys = self.w_assoc = self.rec_loss = self.batch_size = None

    def prior(self, batch_size):
        self.batch_size = batch_size
        self.saved_keys = [[] for i in range(self.batch_size)]
        self.rec_loss = 0.0
        self.w_assoc = torch.zeros((self.batch_size, self.key_size, self.value_size), requires_grad=False, device=device)
        torch.nn.init.normal_(self.w_assoc, mean=0, std=0.1)

    def reset(self, done_task):
        done_task_idx = done_task.view(len(done_task)).nonzero(as_tuple=True)[0]
        tmp = torch.zeros(
            size=(len(done_task_idx), self.key_size, self.value_size),
            requires_grad=False, device=device)
        torch.nn.init.normal_(tmp, mean=0, std=0.1)
        self.w_assoc[done_task_idx] = tmp

    def write(self, brim_hidden, obs, modulation, done_process_mdp):
        key = self.key_encoder(obs)
        value = self.value_encoder(brim_hidden)

        done_process_mdp = done_process_mdp.view(-1).nonzero(as_tuple=True)[0]
        value = modulation * value
        correlation = torch.matmul(key.permute(0, 2, 1), value)
        regularization = torch.matmul(key, key)
        delta_w = self.A * (self.w_max - self.w_assoc) * correlation - self.B * self.w_assoc * regularization
        self.w_assoc[done_process_mdp] = self.w_assoc[done_process_mdp].clone() + delta_w
        avg_modulation = modulation.mean()

        for i in range(len(done_process_mdp)):
            idx = (modulation[i] >= avg_modulation).nonzero(as_tuple=True)[0]
            task_obs = obs[i]
            tmp = list(task_obs[idx.long()])
            self.saved_keys[i].extend(tmp)

    def read(self, query):
        query = self.query_encoder(query).reshape(-1, self.num_head, self.key_size)
        w_assoc = self.w_assoc.clone()
        value = torch.bmm(query, w_assoc)
        value = value.reshape(self.batch_size, self.num_head*self.value_size)
        value = self.value_aggregator(value)
        return value

    def get_done_process(self, done_task):
        samples_query = list()
        done_task = done_task.nonzero(as_tuple=True)[0]
        for i in range(len(done_task)):
            samples_query.append(torch.stack(self.saved_keys[i]))
        key = []
        value = []

        for i in range(len(done_task)):
            done_task_idx = done_task[i]
            with torch.no_grad():
                result = torch.matmul(samples_query[i].unsqueeze(0), self.w_assoc[done_task_idx].unsqueeze(0))
            key.append(samples_query[i].detach().clone())
            value.append(result.squeeze(0).detach().clone())

        return key, value
