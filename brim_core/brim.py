import torch.nn as nn
import torch
from brim_core.blocks import Blocks
from models.encoder import VAERNNEncoder
from utils import helpers as utl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BRIM(nn.Module):
    def __init__(self,
                 # memory
                 use_memory,
                 use_hebb,
                 use_gen,
                 memory_controller_hidden_size,
                 memory_controller_rim_or_gru,
                 memory_key_dim,
                 memory_value_dim,
                 memory_query_dim,
                 # vision core
                 use_stateless_vision_core,
                 # brim
                 use_rim_level1,
                 use_rim_level2,
                 use_rim_level3,
                 rim_top_down_level2_level1,
                 rim_top_down_level3_level2,
                 use_gru_or_rim,
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
                 brim_layers_before_rim_level1,
                 brim_layers_before_rim_level2,
                 brim_layers_before_rim_level3,
                 brim_layers_after_rim_level1,
                 brim_layers_after_rim_level2,
                 brim_layers_after_rim_level3,
                 rim_level1_condition_on_task_inference_latent,
                 rim_level2_condition_on_task_inference_latent,
                 # vae encoder
                 vae_encoder_layers_before_gru,
                 vae_encoder_hidden_size,
                 vae_encoder_layers_after_gru,
                 task_inference_latent_dim,
                 action_dim,
                 action_embed_dim,
                 state_dim,
                 state_embed_dim,
                 reward_size,
                 reward_embed_size,
                 ):
        super(BRIM, self).__init__()

        self.rim_level1_condition_on_task_inference_latent = rim_level1_condition_on_task_inference_latent
        self.rim_level2_condition_on_task_inference_latent = rim_level2_condition_on_task_inference_latent
        self.model = self.initialise_blocks(
                                            use_rim_level1=use_rim_level1,
                                            use_rim_level2=use_rim_level2,
                                            use_rim_level3=use_rim_level3,
                                            rim_top_down_level2_level1=rim_top_down_level2_level1,
                                            rim_top_down_level3_level2=rim_top_down_level3_level2,
                                            use_gru_or_rim=use_gru_or_rim,
                                            rim_level1_hidden_size=rim_level1_hidden_size,
                                            rim_level2_hidden_size=rim_level2_hidden_size,
                                            rim_level3_hidden_size=rim_level3_hidden_size,
                                            rim_level1_output_dim=rim_level1_output_dim,
                                            rim_level2_output_dim=rim_level2_output_dim,
                                            rim_level3_output_dim=rim_level3_output_dim,
                                            rim_level1_num_modules=rim_level1_num_modules,
                                            rim_level2_num_modules=rim_level2_num_modules,
                                            rim_level3_num_modules=rim_level3_num_modules,
                                            rim_level1_topk=rim_level1_topk,
                                            rim_level2_topk=rim_level2_topk,
                                            rim_level3_topk=rim_level3_topk,
                                            brim_layers_before_rim_level1=brim_layers_before_rim_level1,
                                            brim_layers_before_rim_level2=brim_layers_before_rim_level2,
                                            brim_layers_before_rim_level3=brim_layers_before_rim_level3,
                                            brim_layers_after_rim_level1=brim_layers_after_rim_level1,
                                            brim_layers_after_rim_level2=brim_layers_after_rim_level2,
                                            brim_layers_after_rim_level3=brim_layers_after_rim_level3,
                                            rim_level1_condition_on_task_inference_latent=rim_level1_condition_on_task_inference_latent,
                                            rim_level2_condition_on_task_inference_latent=rim_level2_condition_on_task_inference_latent,
                                            task_inference_latent_dim=task_inference_latent_dim,
                                            use_memory=use_memory,
                                            action_dim=action_dim,
                                            reward_dim=reward_size,
                                            action_embed_dim=action_embed_dim,
                                            reward_embed_size=reward_embed_size,
                                            state_embed_dim=state_embed_dim
                                            )

        self.vae_encoder = self.initialise_vae_encoder(vae_encoder_layers_before_gru=vae_encoder_layers_before_gru,
                                                       vae_encoder_hidden_size=vae_encoder_hidden_size,
                                                       vae_encoder_layers_after_gru=vae_encoder_layers_after_gru,
                                                       task_inference_latent_dim=task_inference_latent_dim,
                                                       action_dim=action_dim,
                                                       action_embed_dim=action_embed_dim,
                                                       state_dim=state_dim,
                                                       state_embed_dim=state_embed_dim,
                                                       reward_size=reward_size,
                                                       reward_embed_size=reward_embed_size,
                                                       )

    @staticmethod
    def initialise_blocks(use_rim_level1,
                          use_rim_level2,
                          use_rim_level3,
                          rim_top_down_level2_level1,
                          rim_top_down_level3_level2,
                          use_gru_or_rim,
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
        blocks = Blocks(use_rim_level1=use_rim_level1,
                        use_rim_level2=use_rim_level2,
                        use_rim_level3=use_rim_level3,
                        rim_top_down_level2_level1=rim_top_down_level2_level1,
                        rim_top_down_level3_level2=rim_top_down_level3_level2,
                        use_gru_or_rim=use_gru_or_rim,
                        rim_level1_hidden_size=rim_level1_hidden_size,
                        rim_level2_hidden_size=rim_level2_hidden_size,
                        rim_level3_hidden_size=rim_level3_hidden_size,
                        rim_level1_output_dim=rim_level1_output_dim,
                        rim_level2_output_dim=rim_level2_output_dim,
                        rim_level3_output_dim=rim_level3_output_dim,
                        rim_level1_num_modules=rim_level1_num_modules,
                        rim_level2_num_modules=rim_level2_num_modules,
                        rim_level3_num_modules=rim_level3_num_modules,
                        rim_level1_topk=rim_level1_topk,
                        rim_level2_topk=rim_level2_topk,
                        rim_level3_topk=rim_level3_topk,
                        brim_layers_before_rim_level1=brim_layers_before_rim_level1,
                        brim_layers_before_rim_level2=brim_layers_before_rim_level2,
                        brim_layers_before_rim_level3=brim_layers_before_rim_level3,
                        brim_layers_after_rim_level1=brim_layers_after_rim_level1,
                        brim_layers_after_rim_level2=brim_layers_after_rim_level2,
                        brim_layers_after_rim_level3=brim_layers_after_rim_level3,
                        rim_level1_condition_on_task_inference_latent=rim_level1_condition_on_task_inference_latent,
                        rim_level2_condition_on_task_inference_latent=rim_level2_condition_on_task_inference_latent,
                        task_inference_latent_dim=task_inference_latent_dim,
                        use_memory=use_memory,
                        action_dim=action_dim,
                        reward_dim=reward_dim,
                        action_embed_dim=action_embed_dim,
                        reward_embed_size=reward_embed_size,
                        state_embed_dim=state_embed_dim
                        ).to(device)
        return blocks

    @staticmethod
    def initialise_vae_encoder(vae_encoder_layers_before_gru,
                               vae_encoder_hidden_size,
                               vae_encoder_layers_after_gru,
                               task_inference_latent_dim,
                               action_dim,
                               action_embed_dim,
                               state_dim,
                               state_embed_dim,
                               reward_size,
                               reward_embed_size,
                               ):
        vae_encoder = VAERNNEncoder(
            layers_before_gru=vae_encoder_layers_before_gru,
            hidden_size=vae_encoder_hidden_size,
            layers_after_gru=vae_encoder_layers_after_gru,
            task_inference_latent_dim=task_inference_latent_dim,
            action_dim=action_dim,
            action_embed_dim=action_embed_dim,
            state_dim=state_dim,
            state_embed_dim=state_embed_dim,
            reward_size=reward_size,
            reward_embed_size=reward_embed_size,
        ).to(device)
        return vae_encoder

    def prior(self, batch_size, sample):
        latent_sample, latent_mean, latent_logvar, task_inference_hidden_state = self.vae_encoder.prior(batch_size, sample)
        brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state = self.model.prior(batch_size)

        return (brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state),\
               (latent_sample, latent_mean, latent_logvar, task_inference_hidden_state)

    def reset(self, task_inference_hidden_state, brim_hidden_state, done_task, done_episode):
        task_inference_hidden_state = self.vae_encoder.reset_hidden(task_inference_hidden_state, done_task)
        brim_hidden_state = self.model.reset_hidden(brim_hidden_state, done_task)
        return task_inference_hidden_state, brim_hidden_state

    def forward(self,
                actions,
                states,
                rewards,
                brim_hidden_state,
                task_inference_hidden_state,
                activated_branch,
                sample,
                ):
        brim_outputs1 = []
        brim_outputs2 = []
        brim_outputs3 = []
        brim_outputs4 = []
        brim_outputs5 = []
        brim_hidden_state_output = []

        latent_sample_output = []
        latent_mean_output = []
        latent_logvar_output = []
        task_inference_hidden_state_output = []

        if task_inference_hidden_state.dim() == 3:
            task_inference_hidden_state = task_inference_hidden_state.squeeze(0)

        for idx_step in range(states.shape[0]):
            latent_sample, latent_mean, latent_logvar, task_inference_hidden_state = self.vae_encoder(actions[idx_step],
                                                                                                      states[idx_step],
                                                                                                      rewards[idx_step],
                                                                                                      task_inference_hidden_state,
                                                                                                      sample=sample)
            brim_level3_task_inference_latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                                          add_nonlinearity_to_latent=False,
                                                                          latent_sample=latent_sample,
                                                                          latent_mean=latent_mean,
                                                                          latent_logvar=latent_logvar)
            brim_level1_task_inference_latent = None
            if self.rim_level1_condition_on_task_inference_latent:
                brim_level1_task_inference_latent = utl.get_latent_for_policy(sample_embeddings=False,
                                                                              add_nonlinearity_to_latent=False,
                                                                              latent_sample=latent_sample,
                                                                              latent_mean=latent_mean,
                                                                              latent_logvar=latent_logvar).detach().clone()
            brim_level2_task_inference_latent = None
            if self.rim_level2_condition_on_task_inference_latent:
                brim_level2_task_inference_latent = utl.get_latent_for_policy(sample_embeddings=False,
                                                                              add_nonlinearity_to_latent=False,
                                                                              latent_sample=latent_sample,
                                                                              latent_mean=latent_mean,
                                                                              latent_logvar=latent_logvar).detach().clone()
            brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state = self.model(action=actions[idx_step],
                                                                                                                 state=states[idx_step],
                                                                                                                 reward=rewards[idx_step],
                                                                                                                 brim_hidden_state=brim_hidden_state,
                                                                                                                 brim_level1_task_inference_latent=brim_level1_task_inference_latent,
                                                                                                                 brim_level2_task_inference_latent=brim_level2_task_inference_latent,
                                                                                                                 brim_level3_task_inference_latent=brim_level3_task_inference_latent,
                                                                                                                 activated_branch=activated_branch)
            brim_outputs1.append(brim_output1)
            brim_outputs2.append(brim_output2)
            brim_outputs3.append(brim_output3)
            brim_outputs4.append(brim_output4)
            brim_outputs5.append(brim_output5)
            brim_hidden_state_output.append(brim_hidden_state)

            latent_sample_output.append(latent_sample)
            latent_mean_output.append(latent_mean)
            latent_logvar_output.append(latent_logvar)
            task_inference_hidden_state_output.append(task_inference_hidden_state)

            brim_hidden_state = torch.unsqueeze(brim_hidden_state, 0)

        brim_outputs1 = torch.stack(brim_outputs1)
        brim_outputs2 = torch.stack(brim_outputs2)
        brim_outputs3 = torch.stack(brim_outputs3)
        brim_outputs4 = torch.stack(brim_outputs4)
        brim_outputs5 = torch.stack(brim_outputs5)
        brim_hidden_state_output = torch.stack(brim_hidden_state_output)

        latent_sample_output = torch.stack(latent_sample_output)
        latent_mean_output = torch.stack(latent_mean_output)
        latent_logvar_output = torch.stack(latent_logvar_output)
        task_inference_hidden_state_output = torch.stack(task_inference_hidden_state_output)

        return (brim_outputs1, brim_outputs2, brim_outputs3, brim_outputs4, brim_outputs5, brim_hidden_state_output), brim_hidden_state, \
               (latent_sample_output, latent_mean_output, latent_logvar_output, task_inference_hidden_state_output), task_inference_hidden_state


