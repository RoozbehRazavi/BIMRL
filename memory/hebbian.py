import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Hebbian(nn.Module):
    def __init__(self, num_head, w_max, key_size, value_size, key_encoder_layer, value_encoder_layer, hebb_learning_rate):
        super(Hebbian, self).__init__()
        self.learning_rate = hebb_learning_rate
        self.key_size = key_size
        self.num_head = num_head
        self.value_size = value_size
        self.w_max = w_max
        # hebbian parameters
        A = torch.zeros(size=(1, key_size, value_size), device=device)
        torch.nn.init.normal_(A, mean=0, std=0.01)
        B = torch.zeros(size=(1, value_size, value_size), device=device)
        torch.nn.init.normal_(B, mean=0, std=0.01)
        self.A = nn.Parameter(A)
        self.B = nn.Parameter(B)

        self.key_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(len(key_encoder_layer)):
            self.key_encoder.append(nn.Linear(curr_dim, key_encoder_layer[i]))
            self.key_encoder.append(nn.ReLU())
        self.key_encoder.append(nn.Linear(curr_dim, key_size))
        self.key_encoder = nn.Sequential(*self.key_encoder)

        self.value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(value_encoder_layer)):
            self.value_encoder.append(nn.Linear(curr_dim, value_encoder_layer[i]))
            self.value_encoder.append(nn.ReLU())
        self.value_encoder.append(nn.Linear(curr_dim, value_size))
        self.value_encoder = nn.Sequential(*self.value_encoder)

        self.query_encoder = nn.Linear(key_size, num_head * key_size)
        self.value_aggregator = nn.Linear(num_head * value_size, value_size)

        self.exploitation_w_assoc = self.exploitation_batch_size = self.exploration_w_assoc = self.exploration_batch_size = None

    def prior(self, batch_size, activated_branch):
        if activated_branch == 'exploitation':
            self.exploitation_batch_size = batch_size
            self.exploitation_w_assoc = torch.zeros((self.exploitation_batch_size, self.key_size, self.value_size), requires_grad=False, device=device)
            torch.nn.init.normal_(self.exploitation_w_assoc, mean=0, std=0.1)
        elif activated_branch == 'exploration':
            self.exploration_batch_size = batch_size
            self.exploration_w_assoc = torch.zeros((self.exploration_batch_size, self.key_size, self.value_size), requires_grad=False, device=device)
            torch.nn.init.normal_(self.exploration_w_assoc, mean=0, std=0.1)
        else:
            raise NotImplementedError

    def reset(self, done_task, activated_branch):
        done_task_idx = done_task.view(len(done_task)).nonzero(as_tuple=True)[0]
        tmp = torch.zeros(
            size=(len(done_task_idx), self.key_size, self.value_size),
            requires_grad=False, device=device)
        torch.nn.init.normal_(tmp, mean=0, std=0.1)
        if activated_branch == 'exploration':
            self.exploration_w_assoc[done_task_idx] = tmp
        elif activated_branch == 'exploitation':
            self.exploitation_w_assoc[done_task_idx] = tmp
        else:
            raise NotImplementedError

    def write(self, value, key, modulation, done_process_mdp, activated_branch):
        key = self.key_encoder(key)
        value = self.value_encoder(value)

        done_process_mdp = done_process_mdp.view(-1).nonzero(as_tuple=True)[0]
        batch_size = len(done_process_mdp)
        value = modulation * value
        correlation = torch.bmm(key.permute(0, 2, 1), value)
        regularization = torch.bmm(key.permute(0, 2, 1), key)
        if activated_branch == 'exploration':
            A = self.A.expand(batch_size, -1, -1)
            B = self.B.expand(batch_size, -1, -1)
            a1 = torch.bmm(A, (self.w_max - self.exploration_w_assoc[done_process_mdp].clone()).permute(0, 2, 1))
            a2 = torch.bmm(a1, correlation)
            a3 = torch.bmm(B, self.exploration_w_assoc[done_process_mdp].clone().permute(0, 2, 1))
            a4 = torch.bmm(a3, regularization).permute(0, 2, 1)
            delta_w = a2 - a4
            self.exploration_w_assoc[done_process_mdp] = self.exploration_w_assoc[done_process_mdp].clone() + self.learning_rate * delta_w
        elif activated_branch == 'exploitation':
            A = self.A.expand(batch_size, -1, -1)
            B = self.B.expand(batch_size, -1, -1)
            a1 = torch.bmm(A, (self.w_max - self.exploitation_w_assoc[done_process_mdp].clone()).permute(0, 2, 1))
            a2 = torch.bmm(a1, correlation)
            a3 = torch.bmm(B, self.exploitation_w_assoc[done_process_mdp].clone().permute(0, 2, 1))
            a4 = torch.bmm(a3, regularization).permute(0, 2, 1)
            delta_w = a2 - a4
            self.exploitation_w_assoc[done_process_mdp] = self.exploitation_w_assoc[done_process_mdp].clone() + self.learning_rate * delta_w

    def read(self, query, activated_branch):
        query = self.query_encoder(query).reshape(-1, self.num_head, self.key_size)
        if activated_branch == 'exploration':
            w_assoc = self.exploration_w_assoc.clone()
            batch_size = self.exploration_batch_size
        elif activated_branch == 'exploitation':
            w_assoc = self.exploitation_w_assoc.clone()
            batch_size = self.exploitation_batch_size
        else:
            raise NotImplementedError
        value = torch.bmm(query, w_assoc)
        value = value.reshape(batch_size, self.num_head*self.value_size)
        value = self.value_aggregator(value)
        return value

