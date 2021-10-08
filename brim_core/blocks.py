import torch
import torch.nn as nn
from torch.nn import functional as F
from brim_core.blocks_core_all import BlocksCore
from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Blocks(nn.Module):
    def __init__(self,
                 use_rim_level1,
                 use_rim_level2,
                 use_rim_level3,
                 rim_top_down_level2_level1,
                 rim_top_down_level3_level2,
                 rim_level1_hidden_size,
                 rim_level2_hidden_size,
                 rim_level3_hidden_size,
                 rim_level1_output_dim,
                 rim_level2_output_dim,
                 rim_level3_output_dim,
                 rim_level1_num_modules,
                 rim_level2_num_modules,
                 rim_level3_num_modules,
                 rim_level1_topk,
                 rim_level2_topk,
                 rim_level3_topk,
                 use_gru_or_rim,
                 brim_layers_before_rim_level1,
                 brim_layers_before_rim_level2,
                 brim_layers_before_rim_level3,
                 brim_layers_after_rim_level1,
                 brim_layers_after_rim_level2,
                 brim_layers_after_rim_level3,
                 rim_level1_condition_on_task_inference_latent,
                 rim_level2_condition_on_task_inference_latent,
                 task_inference_latent_dim,
                 use_memory,
                 action_dim,
                 reward_dim,
                 action_embed_dim,
                 reward_embed_size,
                 state_embed_dim
                 ):
        super(Blocks, self).__init__()
        assert rim_top_down_level2_level1 and use_rim_level2
        assert rim_top_down_level3_level2 and use_rim_level3
        assert rim_top_down_level2_level1 and use_gru_or_rim == 'GRU'
        assert rim_top_down_level3_level2 and use_gru_or_rim == 'GRU'

        self.use_rim_level1 = use_rim_level1
        self.use_rim_level2 = use_rim_level2
        self.use_rim_level3 = use_rim_level3
        self.rim_level1_condition_on_task_inference_latent = rim_level1_condition_on_task_inference_latent
        self.rim_level2_condition_on_task_inference_latent = rim_level2_condition_on_task_inference_latent
        self.rim_top_down_level2_level1 = rim_top_down_level2_level1,
        self.rim_top_down_level3_level2 = rim_top_down_level3_level2,
        self.rim_level1_output_dim = rim_level1_output_dim
        self.rim_level2_output_dim = rim_level2_output_dim
        self.rim_level3_output_dim = rim_level3_output_dim
        self.rim_level1_hidden_size = rim_level1_hidden_size
        self.rim_level2_hidden_size = rim_level2_hidden_size
        self.rim_level3_hidden_size = rim_level3_hidden_size

        self.state_encoder = utl.SimpleVision(state_embed_dim)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_dim, reward_embed_size, F.relu)
        brim_input_dim = action_embed_dim + state_embed_dim + reward_embed_size

        if use_rim_level1:
            if len(brim_layers_before_rim_level1) > 0:
                rim_level1_input_dim = brim_layers_before_rim_level1[-1]
            else:
                rim_level1_input_dim = brim_input_dim

            if use_memory:
                rim_level1_input_dim = brim_input_dim + rim_level1_hidden_size

            if rim_level1_condition_on_task_inference_latent:
                rim_level1_input_dim += task_inference_latent_dim * 2

            if rim_top_down_level2_level1:
                rim_level1_input_dim += rim_level2_hidden_size

        if use_rim_level2:

            if len(brim_layers_before_rim_level2) > 0:
                rim_level2_input_dim = brim_layers_before_rim_level2[-1]
            else:
                rim_level2_input_dim = brim_input_dim

            if rim_level2_condition_on_task_inference_latent:
                rim_level2_input_dim += task_inference_latent_dim * 2

            if rim_top_down_level3_level2:
                rim_level2_input_dim += rim_level3_hidden_size

        if use_rim_level3:
            if len(brim_layers_before_rim_level3) > 0:
                rim_level3_input_dim = brim_layers_before_rim_level3[-1]
            else:
                rim_level3_input_dim = brim_input_dim + task_inference_latent_dim

        self.bc_list = self.initialise_rims(use_rim_level1,
                                            use_rim_level2,
                                            use_rim_level3,
                                            rim_top_down_level2_level1,
                                            rim_top_down_level3_level2,
                                            rim_level1_hidden_size,
                                            rim_level2_hidden_size,
                                            rim_level3_hidden_size,
                                            rim_level1_num_modules,
                                            rim_level2_num_modules,
                                            rim_level3_num_modules,
                                            rim_level1_topk,
                                            rim_level2_topk,
                                            rim_level3_topk,
                                            rim_level1_input_dim=rim_level1_input_dim,
                                            rim_level2_input_dim=rim_level2_input_dim,
                                            rim_level3_input_dim=rim_level3_input_dim,
                                            use_gru_or_rim=use_gru_or_rim)
        self.input_embedding_layer_level1,\
        self.input_embedding_layer_level2,\
        self.input_embedding_layer_level3 = self.initialise_input_embedding_layers(brim_input_dim,
                                                                                   brim_layers_before_rim_level1,
                                                                                   brim_layers_before_rim_level2,
                                                                                   brim_layers_before_rim_level3,
                                                                                   use_rim_level1,
                                                                                   use_rim_level2,
                                                                                   use_rim_level3
                                                                                   )
        self.output_layer_level1,\
        self.output_layer_level2,\
        self.output_layer_level3 = self.initialise_output_layers(rim_level1_hidden_size,
                                                                 rim_level2_hidden_size,
                                                                 rim_level3_hidden_size,
                                                                 rim_level1_output_dim,
                                                                 rim_level2_output_dim,
                                                                 rim_level3_output_dim,
                                                                 brim_layers_after_rim_level1,
                                                                 brim_layers_after_rim_level2,
                                                                 brim_layers_after_rim_level3,
                                                                 use_rim_level1,
                                                                 use_rim_level2,
                                                                 use_rim_level3,
                                                                 )

    @staticmethod
    def initialise_rims(use_rim_level1,
                        use_rim_level2,
                        use_rim_level3,
                        rim_top_down_level2_level1,
                        rim_top_down_level3_level2,
                        rim_level1_hidden_size,
                        rim_level2_hidden_size,
                        rim_level3_hidden_size,
                        rim_level1_num_modules,
                        rim_level2_num_modules,
                        rim_level3_num_modules,
                        rim_level1_topk,
                        rim_level2_topk,
                        rim_level3_topk,
                        rim_level1_input_dim,
                        rim_level2_input_dim,
                        rim_level3_input_dim,
                        use_gru_or_rim
                        ):

        blocks_level1 = nn.ModuleList([])
        blocks_level2 = nn.ModuleList([])
        blocks_level3 = None
        if use_rim_level1:
            if use_gru_or_rim == 'GRU':
                blocks_level1.append(nn.GRUCell(rim_level1_input_dim, rim_level1_hidden_size))
                blocks_level1.append(nn.GRUCell(rim_level1_input_dim, rim_level1_hidden_size))
            elif use_gru_or_rim == 'RIM':
                rim_level1_input_dim = rim_level1_input_dim
                if rim_top_down_level2_level1:
                    rim_level1_input_dim += rim_level2_hidden_size

                blocks_level1.append(BlocksCore(ninp=rim_level1_input_dim,
                                                nhid=rim_level1_hidden_size,
                                                num_blocks_in=1,
                                                num_blocks_out=rim_level1_num_modules,
                                                topkval=rim_level1_topk,
                                                do_gru=True,
                                                num_modules_read_input=2,
                                                use_higher=False,
                                                higher_separate_att=True))
                blocks_level1.append(BlocksCore(ninp=rim_level1_input_dim,
                                                nhid=rim_level1_hidden_size,
                                                num_blocks_in=1,
                                                num_blocks_out=rim_level1_num_modules,
                                                topkval=rim_level1_topk,
                                                do_gru=True,
                                                num_modules_read_input=2,
                                                use_higher=False,
                                                higher_separate_att=True))
        if use_rim_level2:
            rim_level2_input_dim = rim_level2_input_dim
            if rim_top_down_level3_level2:
                rim_level2_input_dim += rim_level3_hidden_size

            if use_gru_or_rim == 'GRU':
                blocks_level2.append(nn.GRUCell(rim_level2_input_dim, rim_level2_hidden_size))
                blocks_level2.append(nn.GRUCell(rim_level2_input_dim, rim_level2_hidden_size))
            elif use_gru_or_rim == 'RIM':
                blocks_level1.append(BlocksCore(ninp=rim_level2_input_dim,
                                                nhid=rim_level2_hidden_size,
                                                num_blocks_in=1,
                                                num_blocks_out=rim_level2_num_modules,
                                                topkval=rim_level2_topk,
                                                do_gru=True,
                                                num_modules_read_input=2,
                                                use_higher=False,
                                                higher_separate_att=True))
                blocks_level1.append(BlocksCore(ninp=rim_level2_input_dim,
                                                nhid=rim_level2_hidden_size,
                                                num_blocks_in=1,
                                                num_blocks_out=rim_level2_num_modules,
                                                topkval=rim_level2_topk,
                                                do_gru=True,
                                                num_modules_read_input=2,
                                                use_higher=False,
                                                higher_separate_att=True))
        if use_rim_level3:
            if use_gru_or_rim == 'GRU':
                blocks_level3 = nn.GRUCell(rim_level3_input_dim, rim_level3_hidden_size)
            elif use_gru_or_rim == 'RIM':
                blocks_level3 = BlocksCore(ninp=rim_level3_input_dim,
                                           nhid=rim_level3_hidden_size,
                                           num_blocks_in=1,
                                           num_blocks_out=rim_level3_num_modules,
                                           topkval=rim_level3_topk,
                                           do_gru=True,
                                           num_modules_read_input=2,
                                           use_higher=False,
                                           higher_separate_att=True)
        return blocks_level1, blocks_level2, blocks_level3

    @staticmethod
    def initialise_input_embedding_layers(brim_input_dim,
                                          brim_layers_before_rim_level1,
                                          brim_layers_before_rim_level2,
                                          brim_layers_before_rim_level3,
                                          use_rim_level1,
                                          use_rim_level2,
                                          use_rim_level3,
                                          ):
        level1 = nn.ModuleList()
        level2 = nn.ModuleList()
        level3 = nn.ModuleList()
        if use_rim_level1:
            level1_1 = nn.ModuleList([])
            level1_2 = nn.ModuleList([])
            if len(brim_layers_before_rim_level1) > 0:
                curr_input_dim = brim_input_dim
                for i in range(len(brim_layers_before_rim_level1)):
                    level1_1.append(nn.ReLU())
                    level1_1.append(nn.Linear(curr_input_dim, brim_layers_before_rim_level1[i]))
                    curr_input_dim = brim_layers_before_rim_level1[i]

                curr_input_dim = brim_input_dim
                for i in range(len(brim_layers_before_rim_level1)):
                    level1_1.append(nn.ReLU())
                    level1_2.append(nn.Linear(curr_input_dim, brim_layers_before_rim_level1[i]))
                    curr_input_dim = brim_layers_before_rim_level1[i]
            level1.append(nn.Sequential(*level1_1))
            level1.append(nn.Sequential(*level1_2))

        if use_rim_level2:
            level2_1 = nn.ModuleList([])
            level2_2 = nn.ModuleList([])
            if len(brim_layers_before_rim_level2) > 0:
                curr_input_dim = brim_input_dim
                for i in range(len(brim_layers_before_rim_level2)):
                    level2_1.append(nn.ReLU())
                    level2_1.append(nn.Linear(curr_input_dim, brim_layers_before_rim_level2[i]))
                    curr_input_dim = brim_layers_before_rim_level2[i]

                curr_input_dim = brim_input_dim
                for i in range(len(brim_layers_before_rim_level2)):
                    level2_2.append(nn.ReLU())
                    level2_2.append(nn.Linear(curr_input_dim, brim_layers_before_rim_level2[i]))
                    curr_input_dim = brim_layers_before_rim_level2[i]
            level2.append(nn.Sequential(*level2_1))
            level2.append(nn.Sequential(*level2_2))

        if use_rim_level3:
            if len(brim_layers_before_rim_level3) > 0:
                curr_input_dim = brim_input_dim
                for i in range(len(brim_layers_before_rim_level3)):
                    level1.append(nn.ReLU())
                    level2.append(nn.Linear(curr_input_dim, brim_layers_before_rim_level3[i]))
                    curr_input_dim = brim_layers_before_rim_level3[i]
            level3 = nn.Sequential(*level3)
        return level1, level2, level3

    @staticmethod
    def initialise_output_layers(rim_level1_hidden_size,
                                 rim_level2_hidden_size,
                                 rim_level3_hidden_size,
                                 rim_level1_output_dim,
                                 rim_level2_output_dim,
                                 rim_level3_output_dim,
                                 brim_layers_after_rim_level1,
                                 brim_layers_after_rim_level2,
                                 brim_layers_after_rim_level3,
                                 use_rim_level1,
                                 use_rim_level2,
                                 use_rim_level3,
                                 ):
        level1 = nn.ModuleList([])
        level2 = nn.ModuleList([])
        level3 = nn.ModuleList([])
        if use_rim_level1:
            level1_1 = nn.ModuleList([])
            level1_2 = nn.ModuleList([])
            curr_output_dim = rim_level1_hidden_size
            if len(brim_layers_after_rim_level1) > 0:
                for i in range(len(brim_layers_after_rim_level1)):
                    level1_1.append(nn.Linear(curr_output_dim, brim_layers_after_rim_level1[i]))
                    level1_1.append(nn.ReLU())
                    curr_output_dim = brim_layers_after_rim_level1[i]
            level1_1.append(nn.Linear(curr_output_dim, rim_level1_output_dim))

            curr_output_dim = rim_level1_hidden_size
            if len(brim_layers_after_rim_level1) > 0:
                for i in range(len(brim_layers_after_rim_level1)):
                    level1_2.append(nn.Linear(curr_output_dim, brim_layers_after_rim_level1[i]))
                    level1_2.append(nn.ReLU())
                    curr_output_dim = brim_layers_after_rim_level1[i]
            level1_2.append(nn.Linear(curr_output_dim, rim_level1_output_dim))
            level1.append(nn.Sequential(*level1_1))
            level1.append(nn.Sequential(*level1_2))

        if use_rim_level2:
            level2_1 = nn.ModuleList([])
            level2_2 = nn.ModuleList([])
            curr_output_dim = rim_level2_hidden_size
            if len(brim_layers_after_rim_level2) > 0:
                for i in range(len(brim_layers_after_rim_level2)):
                    level2_1.append(nn.Linear(curr_output_dim, brim_layers_after_rim_level2[i]))
                    level2_1.append(nn.ReLU())
                    curr_output_dim = brim_layers_after_rim_level2[i]
            level2_1.append(nn.Linear(curr_output_dim, rim_level2_output_dim))

            curr_output_dim = rim_level2_hidden_size
            if len(brim_layers_after_rim_level2) > 0:
                for i in range(len(brim_layers_after_rim_level2)):
                    level2_2.append(nn.Linear(curr_output_dim, brim_layers_after_rim_level2[i]))
                    level2_2.append(nn.ReLU())
                    curr_output_dim = brim_layers_after_rim_level2[i]
            level2_2.append(nn.Linear(curr_output_dim, rim_level2_output_dim))
            level2.append(nn.Sequential(*level2_1))
            level2.append(nn.Sequential(*level2_2))

        if use_rim_level3:
            curr_output_dim = rim_level3_hidden_size
            if len(brim_layers_after_rim_level3) > 0:
                for i in range(len(brim_layers_after_rim_level3)):
                    level3.append(nn.Linear(curr_output_dim, brim_layers_after_rim_level3[i]))
                    level3.append(nn.ReLU())
                    curr_output_dim = brim_layers_after_rim_level3[i]
            level3.append(nn.Linear(curr_output_dim, rim_level3_output_dim))

        return level1, level2, level3

    def prior(self, batch_size):
        brim_hidden_state = []
        rim_hidden_size = max(self.rim_level1_hidden_size, self.rim_level2_hidden_size, self.rim_level3_hidden_size)
        weight = next(self.bc_lst[0].block_lstm.parameters())
        brim_hidden_state.append(weight.new_zeros(1, batch_size, 1, rim_hidden_size, requires_grad=True))
        brim_hidden_state.append(weight.new_zeros(1, batch_size, 1, rim_hidden_size, requires_grad=True))
        brim_hidden_state.append(weight.new_zeros(1, batch_size, 1, rim_hidden_size, requires_grad=True))
        brim_hidden_state.append(weight.new_zeros(1, batch_size, 1, rim_hidden_size, requires_grad=True))
        brim_hidden_state.append(weight.new_zeros(1, batch_size, 1, rim_hidden_size, requires_grad=True))

        brim_hidden_state = torch.cat(brim_hidden_state, dim=2)

        h1 = brim_hidden_state[:, :, 0, :self.rim_level1_hidden_size]
        h2 = brim_hidden_state[:, :, 1, :self.rim_level1_hidden_size]
        h3 = brim_hidden_state[:, :, 2, :self.rim_level2_hidden_size]
        h4 = brim_hidden_state[:, :, 3, :self.rim_level2_hidden_size]
        h5 = brim_hidden_state[:, :, 4, :self.rim_level3_hidden_size]

        if self.use_rim_level1:
            brim_output1 = self.output_layer_level1[0](h1)
            brim_output2 = self.output_layer_level1[1](h2)
        else:
            brim_output1 = torch.zeros(size=(1, batch_size, self.rim_level1_output_dim), device=device)
            brim_output2 = torch.zeros(size=(1, batch_size, self.rim_level1_output_dim), device=device)

        if self.use_rim_level2:
            brim_output3 = self.output_layer_level2[0](h3)
            brim_output4 = self.output_layer_level2[1](h4)
        else:
            brim_output3 = torch.zeros(size=(1, batch_size, self.rim_level2_output_dim), device=device)
            brim_output4 = torch.zeros(size=(1, batch_size, self.rim_level2_output_dim), device=device)

        if self.use_rim_level3:
            brim_output5 = self.output_layer_level3(h5)
        else:
            brim_output5 = torch.zeros(size=(1, batch_size, self.rim_level3_output_dim), device=device)

        return brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state

    @staticmethod
    def reset(brim_hidden_state, done_task):
        # brim_hidden_state dim should like (length, batch_size, num_layer, hidden_size)
        if brim_hidden_state.dim() != done_task.dim() + 1:
            if done_task.dim() == 2:
                done = done_task.unsqueeze(0)
            elif done_task.dim() == 1:
                done = done_task.unsqueeze(0).unsqueeze(2)
        brim_hidden_state = brim_hidden_state * (1 - done)
        return brim_hidden_state

    def forward(self,
                action,
                state,
                reward,
                brim_hidden_state,
                brim_level1_task_inference_latent,
                brim_level2_task_inference_latent,
                brim_level3_task_inference_latent,
                activated_branch):
        action = self.action_encoder(action)
        state = self.state_encoder(state)
        reward = self.reward_encoder(reward)
        brim_hidden_state1 = brim_hidden_state[:, :, 0, :self.rim_level1_hidden_size]
        brim_hidden_state2 = brim_hidden_state[:, :, 1, :self.rim_level1_hidden_size]
        brim_hidden_state3 = brim_hidden_state[:, :, 2, :self.rim_level2_hidden_size]
        brim_hidden_state4 = brim_hidden_state[:, :, 3, :self.rim_level2_hidden_size]
        brim_hidden_state5 = brim_hidden_state[:, :, 4, :self.rim_level3_hidden_size]

        brim_input = torch.cat((state, action, reward), dim=-1)
        if activated_branch == 'exploration':
            if self.use_rim_level1:
                level1_input = self.input_embedding_layer_level1[0](brim_input)
                if self.rim_level1_condition_on_task_inference_latent:
                    level1_input = torch.cat((level1_input, brim_level1_task_inference_latent), dim=-1)
                if self.rim_top_down_level2_level1:
                    level1_input = torch.cat((level1_input, brim_hidden_state3), dim=-1)
                brim_hidden_state1 = self.bc_list[0](level1_input, brim_hidden_state1, brim_hidden_state1, idx_layer=0)
                brim_output1 = self.output_layer_level1[0](brim_hidden_state1)
            else:
                brim_output1 = torch.zeros(size=(*brim_hidden_state1.shape[:-1], self.rim_level1_output_dim), device=device)

            if self.use_rim_level2:
                level2_input = self.input_embedding_layer_level2[0](brim_input)
                if self.rim_level2_condition_on_task_inference_latent:
                    level2_input = torch.cat((level2_input, brim_level2_task_inference_latent), dim=-1)
                if self.rim_top_down_level3_level2:
                    level2_input = torch.cat((level2_input, brim_hidden_state5), dim=-1)
                brim_hidden_state3 = self.bc_list[2](level2_input, brim_hidden_state3, brim_hidden_state3, idx_layer=1)
                brim_output3 = self.output_layer_level2[0](brim_hidden_state3)
            else:
                brim_output3 = torch.zeros(size=(*brim_hidden_state3.shape[:-1], self.rim_level2_output_dim), device=device)

            if self.use_rim_level3:
                level3_input = self.input_embedding_layer_level3(brim_input)
                level3_input = torch.cat((level3_input, brim_level3_task_inference_latent), dim=-1)
                brim_hidden_state5 = self.bc_list[4](level3_input, brim_hidden_state5, brim_hidden_state5, idx_layer=2)
                brim_output5 = self.output_layer_level3(brim_hidden_state5)
            else:
                brim_output5 = torch.zeros(size=(*brim_hidden_state5.shape[:-1], self.rim_level3_output_dim), device=device)

            brim_output2 = torch.zeros(size=(*brim_hidden_state2.shape[:-1], self.rim_level1_output_dim), device=device)
            brim_output4 = torch.zeros(size=(*brim_hidden_state4.shape[:-1], self.rim_level2_output_dim), device=device)

        elif activated_branch == 'exploitation':
            if self.use_rim_level1:
                level1_input = self.input_embedding_layer_level1[1](brim_input)
                if self.rim_level1_condition_on_task_inference_latent:
                    level1_input = torch.cat((level1_input, brim_level1_task_inference_latent), dim=-1)
                if self.rim_top_down_level2_level1:
                    level1_input = torch.cat((level1_input, brim_hidden_state4), dim=-1)
                brim_hidden_state2 = self.bc_list[1](level1_input, brim_hidden_state2, brim_hidden_state2, idx_layer=0)
                brim_output2 = self.output_layer_level1[1](brim_hidden_state2)
            else:
                brim_output2 = torch.zeros(size=(*brim_hidden_state2.shape[:-1], self.rim_level1_output_dim), device=device)

            if self.use_rim_level2:
                level2_input = self.input_embedding_layer_level2[1](brim_input)
                if self.rim_level2_condition_on_task_inference_latent:
                    level2_input = torch.cat((level2_input, brim_level2_task_inference_latent), dim=-1)
                if self.rim_top_down_level3_level2:
                    level2_input = torch.cat((level2_input, brim_hidden_state5), dim=-1)
                brim_hidden_state4 = self.bc_list[3](level2_input, brim_hidden_state4, brim_hidden_state4, idx_layer=1)
                brim_output4 = self.output_layer_level2(brim_hidden_state4)
            else:
                brim_output4 = torch.zeros(size=(*brim_hidden_state4.shape[:-1], self.rim_level2_output_dim), device=device)

            if self.use_rim_level3:
                level3_input = self.input_embedding_layer_level3(brim_input)
                level3_input = torch.cat((level3_input, brim_level3_task_inference_latent), dim=-1)
                brim_hidden_state5 = self.bc_list[4](level3_input, brim_hidden_state5, brim_hidden_state5, idx_layer=2)
                brim_output5 = self.output_layer_level3(brim_hidden_state5)
            else:
                brim_output5 = torch.zeros(size=(*brim_hidden_state5.shape[:-1], self.rim_level3_output_dim), device=device)

            brim_output1 = torch.zeros(size=(*brim_hidden_state1.shape[:-1], self.rim_level1_output_dim), device=device)
            brim_output3 = torch.zeros(size=(*brim_hidden_state3.shape[:-1], self.rim_level2_output_dim), device=device)

        elif activated_branch == 'level3':
            if self.use_rim_level3:
                level3_input = self.input_embedding_layer_level3(brim_input)
                level3_input = torch.cat((level3_input, brim_level3_task_inference_latent), dim=-1)
                brim_hidden_state5 = self.bc_list[4](level3_input, brim_hidden_state5, brim_hidden_state5, idx_layer=2)
                brim_output5 = self.output_layer_level3(brim_hidden_state5)
            else:
                brim_output5 = torch.zeros(size=(*brim_hidden_state5.shape[:-1], self.rim_level3_output_dim), device=device)

            brim_output1 = torch.zeros(size=(*brim_hidden_state1.shape[:-1], self.rim_level1_output_dim), device=device)
            brim_output2 = torch.zeros(size=(*brim_hidden_state2.shape[:-1], self.rim_level1_output_dim), device=device)
            brim_output3 = torch.zeros(size=(*brim_hidden_state3.shape[:-1], self.rim_level2_output_dim), device=device)
            brim_output4 = torch.zeros(size=(*brim_hidden_state4.shape[:-1], self.rim_level2_output_dim), device=device)

        else:
            raise NotImplementedError
        rim_hidden_size = max(self.rim_level1_hidden_size, self.rim_level2_hidden_size, self.rim_level3_hidden_size)

        brim_hidden_state1 = torch.cat((brim_hidden_state1, torch.zeros(size=(*brim_hidden_state1.shape[:-1], rim_hidden_size - self.rim_level1_hidden_size), device=device)), dim=-1)
        brim_hidden_state2 = torch.cat((brim_hidden_state2, torch.zeros(size=(*brim_hidden_state2.shape[:-1], rim_hidden_size - self.rim_level1_hidden_size), device=device)), dim=-1)
        brim_hidden_state3 = torch.cat((brim_hidden_state3, torch.zeros(size=(*brim_hidden_state3.shape[:-1], rim_hidden_size - self.rim_level2_hidden_size), device=device)), dim=-1)
        brim_hidden_state4 = torch.cat((brim_hidden_state4, torch.zeros(size=(*brim_hidden_state4.shape[:-1], rim_hidden_size - self.rim_level2_hidden_size), device=device)), dim=-1)
        brim_hidden_state5 = torch.cat((brim_hidden_state5, torch.zeros(size=(*brim_hidden_state5.shape[:-1], rim_hidden_size - self.rim_level3_hidden_size), device=device)), dim=-1)

        brim_hidden_state = torch.cat((brim_hidden_state1, brim_hidden_state2, brim_hidden_state3, brim_hidden_state4, brim_hidden_state5), dim=2)
        return brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state


