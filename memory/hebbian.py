import torch.nn as nn
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Hebbian(nn.Module):
    def __init__(self, args, num_head):
        super(Hebbian, self).__init__()
        self.num_head = num_head
        self.args = args
        self.saved_keys_num = 100
        self.A = 0.5  # nn.parameter.Parameter(torch.ones((1, 1), requires_grad=True))*0.001
        self.B = nn.parameter.Parameter(torch.ones((1, 1), requires_grad=True)) * 0.001
        self.C = nn.parameter.Parameter(torch.ones((1, 1), requires_grad=True)) * 0.001
        self.D = nn.parameter.Parameter(torch.ones((1, 1), requires_grad=True)) * 0.001
        self.E = nn.parameter.Parameter(torch.ones((1, 1), requires_grad=True)) * 0.001
        self.linear_output = torch.nn.Linear(self.num_head * 2 * self.args.brim_hidden_size[0],
                                             2 * self.args.brim_hidden_size[0])
        self.saved_keys = self.w_assoc = self.rec_loss = None

    def prior(self, batch_size):
        self.batch_size = batch_size
        self.saved_keys = [[] for i in range(self.batch_size)]
        self.rec_loss = 0.0
        # TODO appropriate init function
        self.w_assoc = torch.zeros((self.batch_size, self.args.state_embedding_size, self.args.brim_hidden_size[0] * 2),
                                   requires_grad=False, device=device)
        self.w_assoc = self.w_assoc.normal_(0, 0.1)

    def reset(self, done_task):
        done_task_idx = done_task.view(len(done_task)).nonzero(as_tuple=True)[0]
        self.w_assoc[done_task_idx] = torch.zeros(
            size=(len(done_task_idx), self.args.state_embedding_size, self.args.brim_hidden_size[0] * 2),
            requires_grad=False, device=device)

    def write(self, brim_hidden, obs, modulation, done_process_mdp):
        key = obs
        value = brim_hidden
        done_process_mdp = done_process_mdp.view(-1).nonzero(as_tuple=True)[0]
        value = modulation * value
        delta_w = torch.matmul(key.permute(0, 2, 1), value)
        delta_w = self.args.gamma_hebb_memory * self.A * delta_w
        self.w_assoc[done_process_mdp] = self.w_assoc[done_process_mdp].clone() + delta_w
        avg_modulation = modulation.mean()

        for i in range(len(done_process_mdp)):
            idx = (modulation[i] >= avg_modulation).nonzero(as_tuple=True)[0]
            task_obs = obs[i]
            tmp = list(task_obs[idx.long()])
            self.saved_keys[i].extend(tmp)

    def read(self, query):
        w_assoc = self.w_assoc.clone()
        value = torch.bmm(query, w_assoc)
        value = value.reshape(self.batch_size, self.num_head * 2 * self.args.brim_hidden_size[0])
        value = self.linear_output(value)
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
