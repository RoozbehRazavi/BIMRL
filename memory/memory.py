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
                 rim_level1_hidden_size,
                 hebb_learning_rate
                 ):
        super(Hippocampus, self).__init__()
        assert policy_num_steps is not None and max_rollouts_per_task is not None
        self.use_hebb = use_hebb
        self.use_gen = use_gen
        self.read_num_head = read_num_head
        self.combination_num_head = combination_num_head
        self.num_memory_level = 1 + (1 if self.use_hebb else 0) + (1 if self.use_gen else 0)

        self.episodic, self.hebbian = self.initialise_memories(
            use_hebb,
            read_num_head,
            key_size,
            value_size,
            episodic_key_encoder_layer,
            episodic_value_encoder_layer,
            policy_num_steps//max_rollouts_per_task,
            policy_num_steps,
            w_max,
            hebbian_key_encoder_layer,
            hebbian_value_encoder_layer,
            hebb_learning_rate,
            general_key_encoder_layer,
            general_value_encoder_layer,
            general_query_encoder_layer,
            read_memory_to_key_layer,
            read_memory_to_value_layer,
            rim_query_size,
            rim_level1_hidden_size,
            memory_state_embedding,
            state_dim
        )

        self.rim_hidden_to_query, self.read_mha = self.initialise_readout_attention(
            rim_hidden_state_to_query_layers,
            rim_query_size,
            rim_level1_hidden_size,
            value_size,
            combination_num_head)
        self.output = nn.Linear(combination_num_head*value_size, value_size)

    @staticmethod
    def initialise_readout_attention(rim_hidden_state_to_query_layers,
                                     rim_query_size,
                                     rim_level1_hidden_size,
                                     value_size,
                                     combination_num_head
                                     ):
        rim_hidden_to_query = nn.ModuleList([])
        curr_dim = rim_level1_hidden_size
        for i in range(len(rim_hidden_state_to_query_layers)):
            rim_hidden_to_query.append(nn.Linear(curr_dim, rim_hidden_state_to_query_layers[i]))
            # rim_hidden_to_query.append(nn.ReLU())
            curr_dim = rim_hidden_state_to_query_layers[i]
        rim_hidden_to_query.append(nn.Linear(curr_dim, rim_query_size))
        rim_hidden_to_query = nn.Sequential(*rim_hidden_to_query)

        read_mha = nn.MultiheadAttention(
            embed_dim=value_size,
            num_heads=combination_num_head,
            bias=False,
            kdim=rim_query_size,
            vdim=value_size,
            batch_first=True)

        return rim_hidden_to_query, read_mha

    @staticmethod
    def initialise_memories(use_hebb,
                            num_head,
                            key_size,
                            value_size,
                            episodic_key_encoder_layer,
                            episodic_value_encoder_layer,
                            episode_len,
                            policy_num_steps,
                            w_max,
                            hebbian_key_encoder_layer,
                            hebbian_value_encoder_layer,
                            hebb_learning_rate,
                            general_key_encoder_layer,
                            general_value_encoder_layer,
                            general_query_encoder_layer,
                            read_memory_to_key_layer,
                            read_memory_to_value_layer,
                            rim_query_size,
                            rim_level1_hidden_size,
                            memory_state_embedding,
                            state_dim
                            ):

        episodic = DND(
            num_head=num_head,
            key_size=key_size,
            value_size=value_size,
            key_encoder_layer=episodic_key_encoder_layer,
            value_encoder_layer=episodic_value_encoder_layer,
            episode_len=episode_len,
            general_key_encoder_layer=general_key_encoder_layer,
            general_value_encoder_layer=general_value_encoder_layer,
            general_query_encoder_layer=general_query_encoder_layer,
            read_memory_to_key_layer=read_memory_to_key_layer,
            read_memory_to_value_layer=read_memory_to_value_layer,
            rim_query_size=rim_query_size,
            rim_level1_hidden_size=rim_level1_hidden_size,
            memory_state_embedding=memory_state_embedding,
            state_dim=state_dim,
            use_hebb=use_hebb).to(device)

        hebbian = None
        if use_hebb:
            hebbian = Hebbian(
                num_head=num_head,
                w_max=w_max,
                key_size=key_size,
                value_size=value_size,
                key_encoder_layer=hebbian_key_encoder_layer,
                value_encoder_layer=hebbian_value_encoder_layer,
                hebb_learning_rate=hebb_learning_rate,
                rim_query_size=rim_query_size,
                general_query_encoder_layer=general_query_encoder_layer,
                state_dim=state_dim,
                memory_state_embedding=memory_state_embedding,
                read_memory_to_key_layer=read_memory_to_key_layer,
                read_memory_to_value_layer=read_memory_to_value_layer,
            )
        return episodic, hebbian

    def prior(self, batch_size, activated_branch):
        self.episodic.prior(batch_size, activated_branch)

        if self.use_hebb:
            self.hebbian.prior(batch_size, activated_branch)

    def reset(self, done_task, done_episode, activated_branch, A, B):
        self.memory_consolidation(done_task=done_task, done_episode=done_episode, activated_branch=activated_branch, A=A, B=B)

    def read(self, query, rim_hidden_state, activated_branch):
        state, task_inference_latent = query
        task_inference_latent = task_inference_latent.detach()
        epi_k, epi_v = self.episodic.read(state, task_inference_latent, activated_branch)
        if self.use_hebb:
            hebb_k, hebb_v = self.hebbian.read(state, task_inference_latent, activated_branch)
        else:
            hebb_k = hebb_v = torch.zeros(size=(0,), device=device)

        q = self.rim_hidden_to_query(rim_hidden_state)
        k = torch.cat((epi_k, hebb_k), dim=1)
        v = torch.cat((epi_v, hebb_v), dim=1)
        ans = self.read_mha(query=q.unsqueeze(1), key=k, value=v)[0].squeeze(1)
        return ans

    def write(self, key, value, rpe, activated_branch):
        state, task_inf_latent = key
        value = value.detach().clone()
        value.requires_grad = True
        value = value.squeeze(0)
        self.episodic.write(state=state, task_inference_latent=task_inf_latent, value=value, rpe=rpe, activated_branch=activated_branch)

    def memory_consolidation(self, done_task, done_episode, activated_branch, A, B):
        if torch.sum(done_task) > 0 and self.use_hebb:
            self.hebbian.reset(done_task, activated_branch)
        if torch.sum(done_episode) > 0:
            if self.use_hebb:
                done_process_info = self.episodic.get_done_process(done_episode.clone(), activated_branch)
                state = done_process_info[0]
                task_inference_latent = done_process_info[1]
                value = done_process_info[2]
                self.hebbian.write(state=state,
                                   task_inference_latent=task_inference_latent,
                                   value=value,
                                   modulation=done_process_info[3],
                                   done_process_mdp=done_episode, 
                                   activated_branch=activated_branch,
                                   A=A,
                                   B=B)
            self.episodic.reset(done_task=done_task, done_process_mdp=done_episode, activated_branch=activated_branch)

    def compute_intrinsic_reward(self, state, task_inf_latent):
        return self.episodic.compute_intrinsic_reward(state, task_inf_latent)
