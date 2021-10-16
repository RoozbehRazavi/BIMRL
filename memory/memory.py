from memory.episodic import DND
from memory.hebbian import Hebbian
from memory.generative import Generative
import torch.nn as nn
import torch
from memory.helpers import compute_weight, apply_alpha
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO after get value(hidden_state) from memory then apply attention on key query value ( no in each memory module)
# TODO is good have multi head attention on memory_controller ? (test on run)
# TODO remove RIM structure for memory controller
class Hippocampus(nn.Module):
    def __init__(self, args):
        super(Hippocampus, self).__init__()
        self.args = args
        self.num_head = 4
        self.controller_num_head = 4
        self.controller_input_size = 128
        self.fc_layer = [self.controller_input_size]
        self.num_process = self.args.num_processes
        self.memory_controller_hidden_size = self.args.base_memory_controller_hidden_size + \
                                             (
                                              self.args.base_memory_controller_hidden_size if self.args.use_hebb else 0) + \
                                             (self.args.base_memory_controller_hidden_size if self.args.use_gen else 0)

        self.num_memory_level = 1 + (1 if self.args.use_hebb else 0) + (1 if self.args.use_gen else 0)
        self.episodic = DND(args=args, num_head=self.num_head).to(device)
        self.hebbian = Hebbian(args=args, num_head=self.num_head).to(device) if self.args.use_hebb else None
        self.generative = Generative(args=args, num_head=self.num_head) if self.args.use_gen else None
        self.key_encoder = nn.Linear(self.args.state_dim+2*self.args.latent_dim, self.args.state_embedding_size)
        self.value_encoder = nn.Linear(self.args.brim_hidden_size[0], self.args.brim_hidden_size[0])
        self.query_encoder = nn.Linear(self.args.state_dim+self.args.latent_dim*2+self.args.brim_hidden_size[0]*2, self.num_head*self.args.state_embedding_size)
        self.rim_hidden_state_to_key = nn.Linear(self.memory_controller_hidden_size,
                                                 self.args.brim_hidden_size[0]*self.num_memory_level)
        self.brim_hidden_state_to_query = nn.Linear(self.args.brim_hidden_size[0], self.controller_num_head*self.args.brim_hidden_size[0])
        self.value_decoder = nn.Linear(self.args.brim_hidden_size[0], self.args.brim_hidden_size[0])
        self.fc_before_controller = nn.ModuleList([])
        current_input_dim = self.args.state_embedding_size*self.num_head + self.args.state_embedding_size + 2 * \
                            self.args.brim_hidden_size[0] + 2 * self.args.brim_hidden_size[0]

        for i in range(len(self.fc_layer)):
            self.fc_before_controller.append(nn.Linear(current_input_dim, self.fc_layer[i]))
            self.fc_before_controller.append(nn.ReLU())
            current_input_dim = self.fc_layer[i]

        self.controller = nn.GRUCell(self.controller_input_size, self.memory_controller_hidden_size)
        self.linear_output = nn.Linear(self.controller_num_head*self.args.brim_hidden_size[0], self.args.brim_hidden_size[0])
        self.generative_linear_output = torch.nn.Linear(self.num_head * 2 * self.args.brim_hidden_size[0],
                                             2 * self.args.brim_hidden_size[0])
        self.controller_brim_hidden = None
        self.last_task_inf_latent = None
        self.batch_size = None

    def prior(self, batch_size):
        self.episodic.prior(batch_size)

        if self.args.use_hebb:
            self.hebbian.prior(batch_size)

        self.controller_brim_hidden = torch.zeros(size=(batch_size, self.memory_controller_hidden_size),
                                                  requires_grad=True, device=device)
        self.last_task_inf_latent = None
        self.batch_size = batch_size

    def reset(self, done_task, done_episode):
        if not torch.is_tensor(done_task):
            done_task = done_task.copy()
            done_task = torch.tensor(done_task).int().view(len(done_task), 1).double()
        else:
            done_task = done_task.detach().clone()
            done_task = done_task.int().view(len(done_task), 1).double()

        self.memory_consolidation(done_episode=done_episode,
                                  done_task=done_task,
                                  task_inf_latent=self.last_task_inf_latent)

        if self.controller_brim_hidden[0].dim() != done_task.dim():
            if done_task.dim() == 1:
                done_task = done_task.unsqueeze(0)

        self.controller_brim_hidden = (self.controller_brim_hidden * (1 - done_task)).float()

    def update_brim_controller_hidden_state(self, memory_query, saved_key, saved_value, retrieved_brim_hidden1, retrieved_brim_hidden2):
        result = torch.cat(
            (memory_query.reshape(shape=(memory_query.shape[0], self.num_head*self.args.state_embedding_size)), saved_key, saved_value, retrieved_brim_hidden1, retrieved_brim_hidden2), dim=-1)

        for i in range(len(self.fc_layer)):
            result = self.fc_before_controller[i](result)

        _, self.controller_brim_hidden = self.controller(result.unsqueeze(0).detach(), self.controller_brim_hidden)
        self.controller_brim_hidden = self.controller_brim_hidden[0][0]

    def detach_hidden_state(self):
        self.controller_brim_hidden = self.controller_brim_hidden.detach()

    def read(self, query):
        state, task_inf_latent, brim_hidden = query

        if brim_hidden[0].dim() == 3:
            brim_hidden1 = brim_hidden[0].detach().clone().squeeze(0).to(device)
            brim_hidden2 = brim_hidden[1].detach().clone().squeeze(0).to(device)
        else:
            brim_hidden1 = brim_hidden[0].detach().clone().to(device)
            brim_hidden2 = brim_hidden[1].detach().clone().to(device)

        task_inf_latent = task_inf_latent.detach()
        memory_query = self.query_encoder(torch.cat((state, task_inf_latent, brim_hidden1, brim_hidden2), dim=-1))
        memory_query = memory_query.reshape(shape=(state.shape[0], self.num_head, self.args.state_embedding_size))
        epi_result = self.episodic.read(memory_query).split(split_size=self.args.brim_hidden_size[0], dim=-1)

        if self.args.use_hebb:
            hebb_result = self.hebbian.read(memory_query).split(split_size=self.args.brim_hidden_size[0], dim=-1)

        if self.args.use_gen:
            gen_result = self.generative.read(obs_embdd=memory_query, task_id=task_inf_latent)
            gen_result = self.generative_linear_output(gen_result).split(split_size=self.args.brim_hidden_size[0], dim=-1)

        controller = torch.split(F.relu(self.rim_hidden_state_to_key(self.controller_brim_hidden)), self.args.brim_hidden_size[0], dim=-1)
        epi_controller = controller[0]
        hebb_controller = None
        gen_controller = None

        if self.args.use_hebb:
            hebb_controller = controller[1]

        if self.args.use_gen:
            gen_controller = controller[2]

        brim_hidden1 = self.brim_hidden_state_to_query(brim_hidden1).reshape(self.batch_size, self.controller_num_head, self.args.brim_hidden_size[0])
        brim_hidden2 = self.brim_hidden_state_to_query(brim_hidden2).reshape(self.batch_size, self.controller_num_head, self.args.brim_hidden_size[0])
        brim1_weight = compute_weight(epi_controller, hebb_controller, gen_controller, F.relu(brim_hidden1))
        brim2_weight = compute_weight(epi_controller, hebb_controller, gen_controller, F.relu(brim_hidden2))
        value_brim1 = list()
        value_brim2 = list()
        value_brim1.append(epi_result[0].unsqueeze(0))
        value_brim2.append(epi_result[1].unsqueeze(0))

        if self.args.use_hebb:
            value_brim1.append(hebb_result[0].unsqueeze(0))
            value_brim2.append(hebb_result[1].unsqueeze(0))

        if self.args.use_gen:
            value_brim1.append(gen_result[0].unsqueeze(0))
            value_brim2.append(gen_result[1].unsqueeze(0))

        value_brim1 = torch.cat(value_brim1, dim=0)
        value_brim2 = torch.cat(value_brim2, dim=0)
        retrieved_brim_hidden1 = apply_alpha(brim1_weight, value_brim1).reshape(shape=(self.batch_size, self.controller_num_head*self.args.brim_hidden_size[0]))
        retrieved_brim_hidden2 = apply_alpha(brim2_weight, value_brim2).reshape(shape=(self.batch_size, self.controller_num_head*self.args.brim_hidden_size[0]))
        retrieved_brim_hidden1 = F.relu(self.linear_output(retrieved_brim_hidden1))
        retrieved_brim_hidden2 = F.relu(self.linear_output(retrieved_brim_hidden2))

        return memory_query, (retrieved_brim_hidden1, retrieved_brim_hidden2)

    def write(self, key, value, RPE):
        state, task_inf_latent = key
        key_memory = self.key_encoder(torch.cat((state, task_inf_latent), dim=-1))
        value1_memory = value[0].detach().clone()
        value2_memory = value[1].detach().clone()
        value1_memory.requires_grad = True
        value2_memory.requires_grad = True
        value1_memory = self.value_encoder(value1_memory).squeeze(0)
        value2_memory = self.value_encoder(value2_memory).squeeze(0)
        value_memory = torch.cat((value1_memory, value2_memory), dim=-1)
        self.episodic.write(memory_key=key_memory, memory_val=value_memory, RPE=RPE)
        self.last_task_inf_latent = task_inf_latent

        return key_memory, value_memory

    def memory_consolidation(self, done_episode, done_task, task_inf_latent):
        if torch.sum(done_episode) > 0 and self.args.use_hebb:
            done_process_info = self.episodic.get_done_process(done_episode.clone())
            self.hebbian.write(done_process_info[1], done_process_info[0], done_process_info[2], done_episode)
            self.episodic.reset(done_process_mdp=done_episode)

        if torch.sum(done_task) > 0 and self.training and self.args.use_gen:
            done_process_info = self.hebbian.get_done_process(done_task.int().clone())
            self.generative.write(train_data=done_process_info, task_inference_id=task_inf_latent[done_task.nonzero(as_tuple=True)[0]])
            self.hebbian.reset(done_task=done_task)

        return True
