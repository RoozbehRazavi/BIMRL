from memory.episodic import DND
from memory.hebbian import Hebbian
import torch.nn as nn
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Hippocampus(nn.Module):
    def __init__(self,
                 use_hebb,
                 use_gen,
                 read_num_head,
                 combination_num_head,
                 key_size,
                 value_size,
                 policy_num_steps,
                 max_rollouts_per_task,
                 w_max,
                 memory_state_embedding,
                 general_key_encoder_layer,
                 general_value_encoder_layer,
                 general_query_encoder_layer,
                 episodic_key_encoder_layer,
                 episodic_value_encoder_layer,
                 hebbian_key_encoder_layer,
                 hebbian_value_encoder_layer,
                 state_dim,
                 rim_query_size,
                 rim_hidden_state_to_query_layers,
                 read_memory_to_value_layer,
                 read_memory_to_key_layer,
                 rim_level1_hidden_size
                 ):
        super(Hippocampus, self).__init__()
        assert policy_num_steps is not None and max_rollouts_per_task is not None
        self.use_hebb = use_hebb
        self.use_gen = use_gen
        self.read_num_head = read_num_head
        self.combination_num_head = combination_num_head
        self.num_memory_level = 1 + (1 if self.use_hebb else 0) + (1 if self.use_gen else 0)
        self.state_encoder = nn.Linear(state_dim, memory_state_embedding)

        self.episodic, self.hebbian = self.initialise_memories(
            use_hebb,
            read_num_head,
            key_size,
            value_size,
            episodic_key_encoder_layer,
            episodic_value_encoder_layer,
            policy_num_steps//max_rollouts_per_task,
            w_max,
            hebbian_key_encoder_layer,
            hebbian_value_encoder_layer)

        self.key_encoder, self.value_encoder, self.query_encoder = self.initialise_encoder(
            key_size,
            value_size,
            general_key_encoder_layer,
            general_value_encoder_layer,
            general_query_encoder_layer)

        self.rim_hidden_to_query, self.read_memory_to_value, self.read_memory_to_key, self.read_mha = self.initialise_readout_attention(
            rim_hidden_state_to_query_layers,
            read_memory_to_key_layer,
            read_memory_to_value_layer,
            rim_query_size,
            rim_level1_hidden_size,
            value_size,
            combination_num_head)
        self.output = nn.Linear(combination_num_head*value_size, value_size)

    @staticmethod
    def initialise_readout_attention(rim_hidden_state_to_query_layers,
                                     read_memory_to_key_layer,
                                     read_memory_to_value_layer,
                                     rim_query_size,
                                     rim_level1_hidden_size,
                                     value_size,
                                     combination_num_head
                                     ):
        rim_hidden_to_query = nn.ModuleList([])
        curr_dim = rim_level1_hidden_size
        for i in range(len(rim_hidden_state_to_query_layers)):
            rim_hidden_to_query.append(nn.Linear(curr_dim, rim_hidden_state_to_query_layers[i]))
            rim_hidden_to_query.append(nn.ReLU())
        rim_hidden_to_query.append(nn.Linear(curr_dim, rim_query_size))
        rim_hidden_to_query = nn.Sequential(*rim_hidden_to_query)

        read_memory_to_value = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(read_memory_to_value_layer)):
            read_memory_to_value.append(nn.Linear(curr_dim, read_memory_to_value_layer[i]))
            read_memory_to_value.append(nn.ReLU())
        read_memory_to_value.append(nn.Linear(curr_dim, value_size))
        read_memory_to_value = nn.Sequential(*read_memory_to_value)

        read_memory_to_key = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(read_memory_to_key_layer)):
            read_memory_to_key.append(nn.Linear(curr_dim, read_memory_to_key_layer[i]))
            read_memory_to_key.append(nn.ReLU())
        read_memory_to_key.append(nn.Linear(curr_dim, rim_query_size))
        read_memory_to_key = nn.Sequential(*read_memory_to_key)

        read_mha = nn.MultiheadAttention(
            embed_dim=value_size,
            num_heads=combination_num_head,
            bias=False,
            kdim=rim_query_size,
            vdim=value_size,
            batch_first=True)

        return rim_hidden_to_query, read_memory_to_value, read_memory_to_key, read_mha

    @staticmethod
    def initialise_memories(use_hebb,
                            num_head,
                            key_size,
                            value_size,
                            episodic_key_encoder_layer,
                            episodic_value_encoder_layer,
                            episode_len,
                            w_max,
                            hebbian_key_encoder_layer,
                            hebbian_value_encoder_layer
                            ):

        episodic = DND(
            num_head=num_head,
            key_size=key_size,
            value_size=value_size,
            key_encoder_layer=episodic_key_encoder_layer,
            value_encoder_layer=episodic_value_encoder_layer,
            episode_len=episode_len).to(device)

        hebbian = None
        if use_hebb:
            hebbian = Hebbian(
                num_head,
                w_max,
                key_size,
                value_size,
                hebbian_key_encoder_layer,
                hebbian_value_encoder_layer)
        return episodic, hebbian

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
            key_encoder.append(nn.ReLU())
        key_encoder.append(nn.Linear(curr_dim, key_size))
        key_encoder = nn.Sequential(*key_encoder)

        value_encoder = nn.ModuleList([])
        curr_dim = value_size
        for i in range(len(general_value_encoder_layer)):
            value_encoder.append(nn.Linear(curr_dim, general_value_encoder_layer[i]))
            value_encoder.append(nn.ReLU())
        value_encoder.append(nn.Linear(curr_dim, value_size))
        value_encoder = nn.Sequential(*value_encoder)

        query_encoder = nn.ModuleList([])
        curr_dim = key_size
        for i in range(len(general_query_encoder_layer)):
            query_encoder.append(nn.Linear(curr_dim, general_query_encoder_layer[i]))
            query_encoder.append(nn.ReLU())
        query_encoder.append(nn.Linear(curr_dim, key_size))
        query_encoder = nn.Sequential(*query_encoder)
        return key_encoder, value_encoder, query_encoder

    def prior(self, batch_size, activated_branch):
        self.episodic.prior(batch_size, activated_branch)

        if self.use_hebb:
            self.hebbian.prior(batch_size, activated_branch)

    def reset(self, done_episode, activated_branch):
        self.memory_consolidation(done_episode=done_episode, activated_branch=activated_branch)

    def read(self, query, rim_hidden_state, activated_branch):
        state, task_inference_latent = query
        state = self.state_encoder(state)
        task_inference_latent = task_inference_latent.detach()
        memory_query = self.query_encoder(torch.cat((state, task_inference_latent), dim=-1))
        epi_result = self.episodic.read(memory_query, activated_branch).unsqueeze(1)
        ans = epi_result
        if self.use_hebb:
            hebb_result = self.hebbian.read(memory_query, activated_branch).unsqueeze(1)
            ans = torch.cat((epi_result, hebb_result), dim=1)

        q = self.rim_hidden_to_query(rim_hidden_state)
        k = self.read_memory_to_key(ans)
        v = self.read_memory_to_value(ans)
        ans = self.read_mha(query=q.unsqueeze(1), key=k, value=v)[0].squeeze(1)
        return ans

    def write(self, key, value, rpe, activated_branch):
        state, task_inf_latent = key
        state = self.state_encoder(state)
        key_memory = self.key_encoder(torch.cat((state, task_inf_latent), dim=-1))
        value = value.detach().clone()
        value.requires_grad = True
        value = self.value_encoder(value).squeeze(0)
        self.episodic.write(memory_key=key_memory, memory_val=value, rpe=rpe, activated_branch=activated_branch)

    def memory_consolidation(self, done_episode, activated_branch):
        if torch.sum(done_episode) > 0 and self.use_hebb:
            done_process_info = self.episodic.get_done_process(done_episode.clone(), activated_branch)
            self.hebbian.write(done_process_info[1], done_process_info[0], done_process_info[2], done_episode, activated_branch)
            self.episodic.reset(done_process_mdp=done_episode, activated_branch=activated_branch)
        return True
