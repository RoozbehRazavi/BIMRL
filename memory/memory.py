from memory.episodic import DND
from memory.hebbian import Hebbian
import torch.nn as nn
import torch
from memory.helpers import compute_weight, apply_alpha
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO after get value(hidden_state) from memory then apply attention on key query value ( no in each memory module)
# TODO is good have multi head attention on memory_controller ? (test on run)
# TODO remove RIM structure for memory controller
class Hippocampus(nn.Module):
    def __init__(self,
                 use_hebb=False,
                 use_gen=False,
                 read_num_head=4,
                 combination_num_head=4,
                 key_size=16,
                 value_size=16,
                 value_decoder_layers=[32],
                 policy_rim_hidden_state_to_query_layers=[],
                 policy_num_steps=None,
                 max_rollouts_per_task=None,
                 w_max=0.5,
                 memory_state_embedding=0,
                 general_key_encoder_layer=None,
                 general_value_encoder_layer=None,
                 general_query_encoder_layer=None,
                 episodic_key_encoder_layer=None,
                 episodic_value_encoder_layer=None,
                 hebbian_key_encoder_layer=None,
                 hebbian_value_encoder_layer=None,
                 state_dim=None
                 ):
        super(Hippocampus, self).__init__()
        assert policy_num_steps is not None and max_rollouts_per_task is not None
        self.use_hebb = use_hebb
        self.use_gen = use_gen
        self.read_num_head = read_num_head
        self.combination_num_head = combination_num_head
        self.num_memory_level = 1 + (1 if self.args.use_hebb else 0) + (1 if self.args.use_gen else 0)
        self.state_encoder = nn.Linear(state_dim, memory_state_embedding)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads)
        self.episodic = DND(key_size,
                            value_size,
                            episodic_key_encoder_layer,
                            episodic_value_encoder_layer,
                            episodic_query_encoder_layer,
                            policy_num_steps,
                            max_rollouts_per_task).to(device)

        if use_hebb:
            self.hebbian = Hebbian(w_max,
                                   key_size,
                                   value_size,
                                   hebbian_key_encoder_layer,
                                   hebbian_value_encoder_layer,
                                   hebbian_query_encoder_layer)
        else:
            self.hebbian = None

        self.key_encoder = nn.ModuleList([])

        curr_dim = key_size
        for i in range(general_key_encoder_layer):
            self.key_encoder.append(nn.Linear(curr_dim, general_key_encoder_layer[i]))
            self.key_encoder.append(nn.ReLU())
        self.key_encoder.append(nn.Linear(curr_dim, key_size))
        self.key_encoder = nn.Sequential(*self.key_encoder)

        self.value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(general_value_encoder_layer):
            self.value_encoder.append(nn.Linear(curr_dim, general_value_encoder_layer[i]))
            self.value_encoder.append(nn.ReLU())
        self.value_encoder.append(nn.Linear(curr_dim, value_size))
        self.value_encoder = nn.Sequential(*self.value_encoder)

        self.query_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(general_query_encoder_layer):
            self.query_encoder.append(nn.Linear(curr_dim, general_query_encoder_layer[i]))
            self.query_encoder.append(nn.ReLU())
        self.query_encoder.append(nn.Linear(curr_dim, key_size))
        self.query_encoder = nn.Sequential(*self.query_encoder)

        self.last_task_inf_latent = self.batch_size = None

    def prior(self, batch_size):
        self.episodic.prior(batch_size)

        if self.args.use_hebb:
            self.hebbian.prior(batch_size)

        self.last_task_inf_latent = None
        self.batch_size = batch_size

    def reset(self, done_task, done_episode):
        self.memory_consolidation(done_episode=done_episode)

    def read(self, state, task_inference_latent, rim_hidden_state):
        state = self.state_encoder(state)
        task_inference_latent = task_inference_latent.detach()
        memory_query = self.query_encoder(torch.cat((state, task_inference_latent), dim=-1))
        epi_result = self.episodic.read(memory_query)

        if self.args.use_hebb:
            hebb_result = self.hebbian.read(memory_query)

        ans = torch.stack((epi_result, hebb_result), dim=1)

        q = self.final_att_layer_query(rim_hidden_state)
        k = self.final_att_layer_key(ans)
        v = self.final_att_layer_value(ans)
        ans = self.mha(q, k, v)
        return ans

    def write(self, key, value, rpe):
        state, task_inf_latent = key
        key_memory = self.key_encoder(torch.cat((state, task_inf_latent), dim=-1))
        value1_memory = value[0].detach().clone()
        value2_memory = value[1].detach().clone()
        value1_memory.requires_grad = True
        value2_memory.requires_grad = True
        value1_memory = self.value_encoder(value1_memory).squeeze(0)
        value2_memory = self.value_encoder(value2_memory).squeeze(0)
        value_memory = torch.cat((value1_memory, value2_memory), dim=-1)
        self.episodic.write(memory_key=key_memory, memory_val=value_memory, rpe=rpe)
        self.last_task_inf_latent = task_inf_latent

    def memory_consolidation(self, done_task):
        if torch.sum(done_episode) > 0 and self.args.use_hebb:
            done_process_info = self.episodic.get_done_process(done_episode.clone())
            self.hebbian.write(done_process_info[1], done_process_info[0], done_process_info[2], done_episode)
            self.episodic.reset(done_process_mdp=done_episode)
        return True
