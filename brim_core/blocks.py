import torch
import torch.nn as nn
from torch.nn import functional as F
from brim_core.blocks_core_all import BlocksCore
from utils import helpers as utl
from brim_core.brim_new_impl import BRIMCell
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
                 state_embed_dim,
                 new_impl,
                 vae_loss_throughout_vae_encoder_from_rim_level3,
                 residual_task_inference_latent
                 ):
        super(Blocks, self).__init__()
        assert (rim_top_down_level2_level1 and use_rim_level2) or not rim_top_down_level2_level1
        assert (rim_top_down_level3_level2 and use_rim_level3) or not rim_top_down_level3_level2

        self.use_rim_level1 = use_rim_level1
        self.use_rim_level2 = use_rim_level2
        self.use_rim_level3 = use_rim_level3
        self.rim_level1_condition_on_task_inference_latent = rim_level1_condition_on_task_inference_latent
        self.rim_level2_condition_on_task_inference_latent = rim_level2_condition_on_task_inference_latent
        self.rim_top_down_level2_level1 = rim_top_down_level2_level1
        self.rim_top_down_level3_level2 = rim_top_down_level3_level2
        self.rim_level1_output_dim = rim_level1_output_dim
        self.rim_level2_output_dim = rim_level2_output_dim
        self.rim_level3_output_dim = rim_level3_output_dim
        self.rim_level1_hidden_size = rim_level1_hidden_size
        self.rim_level2_hidden_size = rim_level2_hidden_size
        self.rim_level3_hidden_size = rim_level3_hidden_size
        self.state_embed_dim = state_embed_dim
        self.use_gru_or_rim = use_gru_or_rim
        self.new_impl = new_impl
        self.vae_loss_throughout_vae_encoder_from_rim_level3 = vae_loss_throughout_vae_encoder_from_rim_level3
        self.residual_task_inference_latent = residual_task_inference_latent

        self.state_encoder = utl.SimpleVision(state_embed_dim)
        self.action_encoder = utl.FeatureExtractor(action_dim, action_embed_dim, F.relu)
        self.reward_encoder = utl.FeatureExtractor(reward_dim, reward_embed_size, F.relu)
        brim_input_dim = action_embed_dim + state_embed_dim + reward_embed_size
        self.use_fix_dim_level1 = False
        self.use_fix_dim_level2 = False

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
                if rim_level1_input_dim < rim_level2_hidden_size:
                    tmp = brim_layers_before_rim_level1[-1] if len(brim_layers_before_rim_level1) > 0 else brim_input_dim
                    brim_layers_before_rim_level1.append(rim_level2_hidden_size - (rim_level1_input_dim - tmp))
                    rim_level1_input_dim = rim_level2_hidden_size
                else:
                    # TODO unfortunately should add Linear Layer
                    self.use_fix_dim_level1 = True
                    self.rim1_input_fix_dim = nn.ModuleList([])
                    self.rim1_input_fix_dim.append(nn.Linear(rim_level2_hidden_size, rim_level1_input_dim))
                    self.rim1_input_fix_dim.append(nn.Linear(rim_level2_hidden_size, rim_level1_input_dim))

        else:
            rim_level1_input_dim = 0

        if use_rim_level2:
            if len(brim_layers_before_rim_level2) > 0:
                rim_level2_input_dim = brim_layers_before_rim_level2[-1]
            else:
                rim_level2_input_dim = brim_input_dim

            if rim_level2_condition_on_task_inference_latent:
                rim_level2_input_dim += task_inference_latent_dim * 2

            if rim_top_down_level3_level2:
                if rim_level2_input_dim < rim_level3_hidden_size:
                    tmp = brim_layers_before_rim_level2[-1] if len(brim_layers_before_rim_level2) > 0 else brim_input_dim
                    brim_layers_before_rim_level2.append(rim_level3_hidden_size - (rim_level2_input_dim - tmp))
                    rim_level2_input_dim = rim_level3_hidden_size
                else:
                    self.use_fix_dim_level2 = True
                    self.rim2_input_fix_dim = nn.ModuleList([])
                    self.rim2_input_fix_dim.append(nn.Linear(rim_level3_hidden_size, rim_level2_input_dim))
                    self.rim2_input_fix_dim.append(nn.Linear(rim_level3_hidden_size, rim_level2_input_dim))
        else:
            rim_level2_input_dim = 0

        if use_rim_level3:
            if len(brim_layers_before_rim_level3) > 0:
                rim_level3_input_dim = brim_layers_before_rim_level3[-1]
            else:
                rim_level3_input_dim = brim_input_dim
            rim_level3_input_dim += task_inference_latent_dim
        else:
            rim_level3_input_dim = 0

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
                                            use_gru_or_rim=use_gru_or_rim,
                                            new_impl=new_impl)
        self.input_embedding_layer_level1,\
        self.input_embedding_layer_level2,\
        self.input_embedding_layer_level3 = self.initialise_input_embedding_layers(brim_input_dim,
                                                                                   brim_layers_before_rim_level1,
                                                                                   brim_layers_before_rim_level2,
                                                                                   brim_layers_before_rim_level3,
                                                                                   use_rim_level1,
                                                                                   use_rim_level2,
                                                                                   use_rim_level3,
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
                        use_gru_or_rim,
                        new_impl
                        ):
        if new_impl:
            assert rim_level1_hidden_size%rim_level1_num_modules == 0
            assert rim_level2_hidden_size % rim_level2_num_modules == 0
            assert rim_level3_hidden_size % rim_level3_num_modules == 0

        blocks_level1 = nn.ModuleList([])
        blocks_level2 = nn.ModuleList([])
        blocks_level3 = None
        if use_rim_level1:
            if use_gru_or_rim == 'GRU':
                blocks_level1.append(nn.GRUCell(rim_level1_input_dim, rim_level1_hidden_size))
                blocks_level1.append(nn.GRUCell(rim_level1_input_dim, rim_level1_hidden_size))
            elif use_gru_or_rim == 'RIM':
                if new_impl:
                    blocks_level1.append(BRIMCell(
                        device=device,
                        use_higher=rim_top_down_level2_level1,
                        input_size=rim_level1_input_dim,
                        hidden_size=int(rim_level1_hidden_size/rim_level1_num_modules),
                        num_units=rim_level1_num_modules,
                        k=rim_level1_topk,
                        rnn_cell='GRU',
                        input_dropout=0.0,
                        comm_dropout=0.0,
                    ))
                    blocks_level1.append(BRIMCell(
                        device=device,
                        use_higher=rim_top_down_level2_level1,
                        input_size=rim_level1_input_dim,
                        hidden_size=int(rim_level1_hidden_size/rim_level1_num_modules),
                        num_units=rim_level1_num_modules,
                        k=rim_level1_topk,
                        rnn_cell='GRU',
                        input_dropout=0.0,
                        comm_dropout=0.0,
                    ))
                else:
                    blocks_level1.append(BlocksCore(ninp=rim_level1_input_dim,
                                                    nhid=rim_level1_hidden_size,
                                                    num_blocks_in=1,
                                                    num_blocks_out=rim_level1_num_modules,
                                                    topkval=rim_level1_topk,
                                                    do_gru=True,
                                                    num_modules_read_input=2,
                                                    use_higher=rim_top_down_level2_level1,
                                                    higher_separate_att=True))
                    blocks_level1.append(BlocksCore(ninp=rim_level1_input_dim,
                                                    nhid=rim_level1_hidden_size,
                                                    num_blocks_in=1,
                                                    num_blocks_out=rim_level1_num_modules,
                                                    topkval=rim_level1_topk,
                                                    do_gru=True,
                                                    num_modules_read_input=2,
                                                    use_higher=rim_top_down_level2_level1,
                                                    higher_separate_att=True))
        if use_rim_level2:
            rim_level2_input_dim = rim_level2_input_dim
            if use_gru_or_rim == 'GRU':
                blocks_level2.append(nn.GRUCell(rim_level2_input_dim, rim_level2_hidden_size))
                blocks_level2.append(nn.GRUCell(rim_level2_input_dim, rim_level2_hidden_size))
            elif use_gru_or_rim == 'RIM':
                if new_impl:
                    blocks_level2.append(BRIMCell(
                        device=device,
                        use_higher=rim_top_down_level3_level2,
                        input_size=rim_level2_input_dim,
                        hidden_size=int(rim_level2_hidden_size / rim_level2_num_modules),
                        num_units=rim_level2_num_modules,
                        k=rim_level2_topk,
                        rnn_cell='GRU',
                        input_dropout=0.0,
                        comm_dropout=0.0
                    ))
                    blocks_level2.append(BRIMCell(
                        device=device,
                        use_higher=rim_top_down_level3_level2,
                        input_size=rim_level2_input_dim,
                        hidden_size=int(rim_level2_hidden_size / rim_level2_num_modules),
                        num_units=rim_level2_num_modules,
                        k=rim_level2_topk,
                        rnn_cell='GRU',
                        input_dropout=0.0,
                        comm_dropout=0.0
                    ))
                else:
                    blocks_level2.append(BlocksCore(ninp=rim_level2_input_dim,
                                                    nhid=rim_level2_hidden_size,
                                                    num_blocks_in=1,
                                                    num_blocks_out=rim_level2_num_modules,
                                                    topkval=rim_level2_topk,
                                                    do_gru=True,
                                                    num_modules_read_input=2,
                                                    use_higher=rim_top_down_level3_level2,
                                                    higher_separate_att=True))
                    blocks_level2.append(BlocksCore(ninp=rim_level2_input_dim,
                                                    nhid=rim_level2_hidden_size,
                                                    num_blocks_in=1,
                                                    num_blocks_out=rim_level2_num_modules,
                                                    topkval=rim_level2_topk,
                                                    do_gru=True,
                                                    num_modules_read_input=2,
                                                    use_higher=rim_top_down_level3_level2,
                                                    higher_separate_att=True))
        if use_rim_level3:
            if use_gru_or_rim == 'GRU':
                blocks_level3 = nn.GRUCell(rim_level3_input_dim, rim_level3_hidden_size)
            elif use_gru_or_rim == 'RIM':
                if new_impl:
                    blocks_level3 = BRIMCell(
                        device=device,
                        use_higher=False,
                        input_size=rim_level3_input_dim,
                        hidden_size=int(rim_level3_hidden_size / rim_level3_num_modules),
                        num_units=rim_level3_num_modules,
                        k=rim_level2_topk,
                        rnn_cell='GRU',
                        input_dropout=0.0,
                        comm_dropout=0.0
                    )
                else:
                    blocks_level3 = BlocksCore(ninp=rim_level3_input_dim,
                                               nhid=rim_level3_hidden_size,
                                               num_blocks_in=1,
                                               num_blocks_out=rim_level3_num_modules,
                                               topkval=rim_level3_topk,
                                               do_gru=True,
                                               num_modules_read_input=2,
                                               use_higher=False,
                                               higher_separate_att=True)
            blocks_level3 = blocks_level3.to(device)

        return blocks_level1.to(device), blocks_level2.to(device), blocks_level3

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
                    level1_2.append(nn.ReLU())
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
                    level3.append(nn.ReLU())
                    level3.append(nn.Linear(curr_input_dim, brim_layers_before_rim_level3[i]))
                    curr_input_dim = brim_layers_before_rim_level3[i]
            level3 = nn.Sequential(*level3)
        return level1.to(device), level2.to(device), level3.to(device)

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
            level3 = nn.Sequential(*level3)

        return level1.to(device), level2.to(device), level3.to(device)

    def prior(self, batch_size):
        brim_hidden_state = []
        rim_hidden_size = max(self.rim_level1_hidden_size, self.rim_level2_hidden_size, self.rim_level3_hidden_size)
        brim_hidden_state.append(torch.zeros((1, batch_size, 1, rim_hidden_size), requires_grad=True, device=device))
        brim_hidden_state.append(torch.zeros((1, batch_size, 1, rim_hidden_size), requires_grad=True, device=device))
        brim_hidden_state.append(torch.zeros((1, batch_size, 1, rim_hidden_size), requires_grad=True, device=device))
        brim_hidden_state.append(torch.zeros((1, batch_size, 1, rim_hidden_size), requires_grad=True, device=device))
        brim_hidden_state.append(torch.zeros((1, batch_size, 1, rim_hidden_size), requires_grad=True, device=device))

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
    def reset_hidden(brim_hidden_state, done):
        # brim_hidden_state dim should like (length, batch_size, num_layer, hidden_size)
        if brim_hidden_state.dim() != done.dim():
            if done.dim() == 2:
                done = done.unsqueeze(-1)
            elif done.dim() == 1:
                done = done.unsqueeze(0).unsqueeze(2)
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
        batch_size = state.shape[0]
        tmp_state = state.clone()
        state = state.view(-1, state.shape[-1])
        state = utl.image_obs(state)
        state = self.state_encoder(state)
        tmp1 = state.view(batch_size, self.state_embed_dim - 1)
        tmp2 = tmp_state[:, -2:-1]
        state = torch.cat((tmp1, tmp2), dim=-1)
        reward = self.reward_encoder(reward)
        if brim_hidden_state.dim() == 4:
            brim_hidden_state = brim_hidden_state.squeeze(0)
        brim_hidden_state1 = brim_hidden_state[:, 0, :self.rim_level1_hidden_size]
        brim_hidden_state2 = brim_hidden_state[:, 1, :self.rim_level1_hidden_size]
        brim_hidden_state3 = brim_hidden_state[:, 2, :self.rim_level2_hidden_size]
        brim_hidden_state4 = brim_hidden_state[:, 3, :self.rim_level2_hidden_size]
        brim_hidden_state5 = brim_hidden_state[:, 4, :self.rim_level3_hidden_size]

        brim_input = torch.cat((state, action, reward), dim=-1)
        if activated_branch == 'exploration':
            if self.use_rim_level1:
                level1_input = self.input_embedding_layer_level1[0](brim_input)
                if self.rim_level1_condition_on_task_inference_latent:
                    level1_input = torch.cat((level1_input, brim_level1_task_inference_latent), dim=-1)
                if self.use_fix_dim_level1:
                    brim_hidden_state3_ = self.rim1_input_fix_dim[0](brim_hidden_state3.detach())
                else:
                    brim_hidden_state3_ = brim_hidden_state3.detach()
                if self.rim_top_down_level2_level1:
                    brim_hidden_state1 = [brim_hidden_state1, brim_hidden_state3_]
                else:
                    brim_hidden_state1 = [brim_hidden_state1]
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state1, _ = self.bc_list[0][0](level1_input, brim_hidden_state3_, brim_hidden_state1[0])
                    else:
                        brim_hidden_state1, _ = self.bc_list[0][0](level1_input, brim_hidden_state1, brim_hidden_state1, idx_layer=0)
                else:
                    brim_hidden_state1 = self.bc_list[0][0](level1_input, brim_hidden_state1)
                brim_output1 = self.output_layer_level1[0](brim_hidden_state1)
            else:
                brim_output1 = torch.zeros(size=(*brim_hidden_state1.shape[:-1], self.rim_level1_output_dim), device=device)

            if self.use_rim_level2:
                level2_input = self.input_embedding_layer_level2[0](brim_input)
                if self.rim_level2_condition_on_task_inference_latent:
                    level2_input = torch.cat((level2_input, brim_level2_task_inference_latent), dim=-1)
                if self.use_fix_dim_level2:
                    brim_hidden_state5_ = self.rim2_input_fix_dim[0](brim_hidden_state5.detach())
                else:
                    brim_hidden_state5_ = brim_hidden_state5.detach()

                if self.rim_top_down_level3_level2:
                    brim_hidden_state3 = [brim_hidden_state3, brim_hidden_state5_]
                else:
                    brim_hidden_state3 = [brim_hidden_state3]
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state3, _ = self.bc_list[1][0](level2_input, brim_hidden_state5_, brim_hidden_state3[0])
                    else:
                        brim_hidden_state3, _ = self.bc_list[1][0](level2_input, brim_hidden_state3, brim_hidden_state3, idx_layer=0)
                else:
                    brim_hidden_state3 = self.bc_list[1][0](level2_input, brim_hidden_state3)
                brim_output3 = self.output_layer_level2[0](brim_hidden_state3)
            else:
                brim_output3 = torch.zeros(size=(*brim_hidden_state3.shape[:-1], self.rim_level2_output_dim), device=device)

            if self.use_rim_level3:
                level3_input = self.input_embedding_layer_level3(brim_input)
                if not self.vae_loss_throughout_vae_encoder_from_rim_level3:
                    assert self.residual_task_inference_latent
                    level3_input = torch.cat((level3_input, brim_level3_task_inference_latent.detach()), dim=-1)
                else:
                    level3_input = torch.cat((level3_input, brim_level3_task_inference_latent), dim=-1)
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state5, _ = self.bc_list[2](level3_input, None, brim_hidden_state5)
                    else:
                        brim_hidden_state5, _ = self.bc_list[2](level3_input, [brim_hidden_state5], [brim_hidden_state5], idx_layer=0)
                else:
                    brim_hidden_state5 = self.bc_list[2](level3_input, brim_hidden_state5)
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
                if self.use_fix_dim_level1:
                    brim_hidden_state4_ = self.rim1_input_fix_dim[1](brim_hidden_state4.detach())
                else:
                    brim_hidden_state4_ = brim_hidden_state4.detach()
                if self.rim_top_down_level2_level1:
                    brim_hidden_state2 = [brim_hidden_state2, brim_hidden_state4_]
                else:
                    brim_hidden_state2 = [brim_hidden_state2]
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state2, _ = self.bc_list[0][1](level1_input, brim_hidden_state4_, brim_hidden_state2[0])
                    else:
                        brim_hidden_state2, _ = self.bc_list[0][1](level1_input, brim_hidden_state2, brim_hidden_state2, idx_layer=0)
                else:
                    brim_hidden_state2 = self.bc_list[0][1](level1_input, brim_hidden_state2)
                brim_output2 = self.output_layer_level1[1](brim_hidden_state2)
            else:
                brim_output2 = torch.zeros(size=(*brim_hidden_state2.shape[:-1], self.rim_level1_output_dim), device=device)

            if self.use_rim_level2:
                level2_input = self.input_embedding_layer_level2[1](brim_input)
                if self.rim_level2_condition_on_task_inference_latent:
                    level2_input = torch.cat((level2_input, brim_level2_task_inference_latent), dim=-1)
                if self.use_fix_dim_level2:
                    brim_hidden_state5_ = self.rim2_input_fix_dim[1](brim_hidden_state5.detach())
                else:
                    brim_hidden_state5_ = brim_hidden_state5.detach()
                if self.rim_top_down_level3_level2:
                    brim_hidden_state4 = [brim_hidden_state4, brim_hidden_state5_]
                else:
                    brim_hidden_state4 = [brim_hidden_state4]
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state4, _ = self.bc_list[1][1](level2_input, brim_hidden_state5_, brim_hidden_state4[0])
                    else:
                        brim_hidden_state4, _ = self.bc_list[1][1](level2_input, brim_hidden_state4, brim_hidden_state4, idx_layer=0)
                else:
                    brim_hidden_state4 = self.bc_list[1][1](level2_input, brim_hidden_state4)
                brim_output4 = self.output_layer_level2[1](brim_hidden_state4)
            else:
                brim_output4 = torch.zeros(size=(*brim_hidden_state4.shape[:-1], self.rim_level2_output_dim), device=device)

            if self.use_rim_level3:
                level3_input = self.input_embedding_layer_level3(brim_input)
                if not self.vae_loss_throughout_vae_encoder_from_rim_level3:
                    assert self.residual_task_inference_latent
                    level3_input = torch.cat((level3_input, brim_level3_task_inference_latent.detach()), dim=-1)
                else:
                    level3_input = torch.cat((level3_input, brim_level3_task_inference_latent), dim=-1)
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state5, _ = self.bc_list[2](x1=level3_input, x2=None, hs=brim_hidden_state5)
                    else:
                        brim_hidden_state5, _ = self.bc_list[2](level3_input, [brim_hidden_state5], [brim_hidden_state5], idx_layer=0)
                else:
                    brim_hidden_state5 = self.bc_list[2](level3_input, brim_hidden_state5)
                brim_output5 = self.output_layer_level3(brim_hidden_state5)
            else:
                brim_output5 = torch.zeros(size=(*brim_hidden_state5.shape[:-1], self.rim_level3_output_dim), device=device)

            brim_output1 = torch.zeros(size=(*brim_hidden_state1.shape[:-1], self.rim_level1_output_dim), device=device)
            brim_output3 = torch.zeros(size=(*brim_hidden_state3.shape[:-1], self.rim_level2_output_dim), device=device)

        elif activated_branch == 'level3':
            if self.use_rim_level3:
                level3_input = self.input_embedding_layer_level3(brim_input)
                if not self.vae_loss_throughout_vae_encoder_from_rim_level3:
                    assert self.residual_task_inference_latent
                    level3_input = torch.cat((level3_input, brim_level3_task_inference_latent.detach()), dim=-1)
                else:
                    level3_input = torch.cat((level3_input, brim_level3_task_inference_latent), dim=-1)
                if self.use_gru_or_rim == 'RIM':
                    if self.new_impl:
                        brim_hidden_state5, _ = self.bc_list[2](x1=level3_input, x2=None, hs=brim_hidden_state5)
                    else:
                        brim_hidden_state5, _ = self.bc_list[2](level3_input, [brim_hidden_state5], [brim_hidden_state5], idx_layer=0)
                else:
                    brim_hidden_state5 = self.bc_list[2](level3_input, brim_hidden_state5)
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

        brim_hidden_state = torch.stack((brim_hidden_state1, brim_hidden_state2, brim_hidden_state3, brim_hidden_state4, brim_hidden_state5), dim=1)
        return brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state


