import torch.nn as nn
import torch
from brim_core.brim import BRIM
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BRIMCore(nn.Module):
    def __init__(self,
                 # brim core ablation
                 use_memory,
                 use_hebb,
                 use_gen,
                 use_stateful_vision_core,
                 use_rim_level1,
                 use_rim_level2,
                 use_rim_level3,
                 rim_top_down_level2_level1,
                 rim_top_down_level3_level2,
                 # brim
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
                 new_impl,
                 vae_loss_throughout_vae_encoder_from_rim_level3,
                 residual_task_inference_latent,
                 rim_output_size_to_vision_core,
                 memory_params,
                 ):
        super(BRIMCore, self).__init__()
        assert (not use_memory and not use_hebb and not use_gen) or use_memory
        assert (use_gru_or_rim == 'GRU' and rim_level1_num_modules == 1 and rim_level2_num_modules == 1 and rim_level3_num_modules == 1) or use_gru_or_rim == 'RIM'
        assert rim_level1_topk <= rim_level1_num_modules and rim_level2_topk <= rim_level2_num_modules and rim_level3_topk <= rim_level3_num_modules
        self.use_rim_level1 = use_rim_level1
        self.use_rim_level2 = use_rim_level2
        self.use_rim_level3 = use_rim_level3
        self.use_stateful_vision_core = use_stateful_vision_core

        self.brim = self.initialise_brim(use_memory,
                                         memory_params,
                                         use_stateful_vision_core,
                                         use_rim_level1,
                                         use_rim_level2,
                                         use_rim_level3,
                                         rim_top_down_level2_level1,
                                         rim_top_down_level3_level2,
                                         # brim
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
                                         new_impl,
                                         vae_loss_throughout_vae_encoder_from_rim_level3,
                                         residual_task_inference_latent,
                                         rim_output_size_to_vision_core
                                         )
        for name, param in self.brim.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    @staticmethod
    def initialise_brim(use_memory,
                        memory_params,
                        use_stateful_vision_core,
                        use_rim_level1,
                        use_rim_level2,
                        use_rim_level3,
                        rim_top_down_level2_level1,
                        rim_top_down_level3_level2,
                        # brim
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
                        new_impl,
                        vae_loss_throughout_vae_encoder_from_rim_level3,
                        residual_task_inference_latent,
                        rim_output_size_to_vision_core
                        ):
        brim = BRIM(use_memory=use_memory,
                    memory_params=memory_params,
                    use_stateful_vision_core=use_stateful_vision_core,
                    use_rim_level1=use_rim_level1,
                    use_rim_level2=use_rim_level2,
                    use_rim_level3=use_rim_level3,
                    rim_top_down_level2_level1=rim_top_down_level2_level1,
                    rim_top_down_level3_level2=rim_top_down_level3_level2,
                    # brim
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
                    # vae encoder
                    vae_encoder_layers_before_gru=vae_encoder_layers_before_gru,
                    vae_encoder_hidden_size=vae_encoder_hidden_size,
                    vae_encoder_layers_after_gru=vae_encoder_layers_after_gru,
                    task_inference_latent_dim=task_inference_latent_dim,
                    action_dim=action_dim,
                    action_embed_dim=action_embed_dim,
                    state_dim=state_dim,
                    state_embed_dim=state_embed_dim,
                    reward_size=reward_size,
                    reward_embed_size=reward_embed_size,
                    new_impl=new_impl,
                    vae_loss_throughout_vae_encoder_from_rim_level3=vae_loss_throughout_vae_encoder_from_rim_level3,
                    residual_task_inference_latent=residual_task_inference_latent,
                    rim_output_size_to_vision_core=rim_output_size_to_vision_core
                    ).to(device)
        return brim

    def _sample_gaussian(self, mu, logvar, num=None):
        if num is None:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            raise NotImplementedError
            std = torch.exp(0.5 * logvar).repeat(num, 1)
            eps = torch.randn_like(std)
            mu = mu.repeat(num, 1)
            return eps.mul(std).add_(mu)

    def reset_hidden(self,
                     policy_state_encoder,
                     task_inference_hidden_state,
                     brim_hidden_state,
                     done_task,
                     done_episode
                     ):
        if self.use_stateful_vision_core:
            policy_state_encoder.state_encoder.reset(done_task)
        return self.brim.reset(task_inference_hidden_state, brim_hidden_state, done_task, done_episode)

    def prior(self,
              policy_network,
              batch_size,
              sample,
              state,
              activated_branch):
        embedd_state = not activated_branch == 'level3'
        if embedd_state and self.use_stateful_vision_core:
            policy_network.state_encoder.prior(batch_size)
        if embedd_state:
            state_process = policy_network.state_process
        else:
            state_process = None
        return self.brim.prior(batch_size=batch_size, sample=sample, state=state, state_process=state_process, embedd_state=embedd_state)

    def forward(self,
                actions,
                states,
                rewards,
                brim_hidden_state,
                task_inference_hidden_state,
                return_prior,
                sample,
                detach_every,
                activated_branch,
                policy,
                prev_state,
                rpe
                ):
        # shape should be: sequence_len x batch_size x hidden_size
        actions = actions.reshape((-1, *actions.shape[-2:]))
        states = states.reshape((-1, *states.shape[-2:]))
        rewards = rewards.reshape((-1, *rewards.shape[-2:]))

        if brim_hidden_state is not None:
            # brim hidden state shape : (length, batch_size, num_layers, hidden_state_size)
            brim_hidden_state = brim_hidden_state.reshape((-1, *brim_hidden_state.shape[-2:]))

        if return_prior:
            # if hidden state is none, start with the prior
            # TODO debug this part for prior state
            (prior_brim_output1, prior_brim_output2, prior_brim_output3, prior_brim_output4,
             prior_brim_output5, prior_brim_hidden_state), \
            (prior_sample, prior_mean, prior_logvar, prior_task_inference_hidden_state), prior_extras_information = self.prior(
                policy_network=policy,
                batch_size=actions.shape[1],
                sample=sample,
                state=prev_state,
                activated_branch=activated_branch)

            brim_hidden_state = prior_brim_hidden_state.clone()
            task_inference_hidden_state = prior_task_inference_hidden_state.clone()
        if policy is not None:
            state_process = policy.state_process
            state_encoder = policy.state_encoder
        else:
            state_process = None
            state_encoder = None
        if detach_every is None:
            (brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_states),  _, \
            (latent_sample, latent_mean, latent_logvar, task_inference_hidden_states), _, extras_information = self.brim(
                actions,
                states,
                rewards,
                brim_hidden_state,
                task_inference_hidden_state,
                activated_branch,
                sample,
                state_process=state_process)
        else:
            brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_states, policy_embedded_state = [], [], [], [], [], [], []
            latent_sample, latent_mean, latent_logvar, task_inference_hidden_states = [], [], [], []
            extras_information = {}
            for i in range(int(np.ceil(states.shape[0] / detach_every))):
                curr_actions = actions[i * detach_every:i * detach_every + detach_every]
                curr_states = states[i * detach_every:i * detach_every + detach_every]
                curr_rewards = rewards[i * detach_every:i * detach_every + detach_every]

                (curr_output1, curr_output2, curr_output3, curr_output4, curr_output5,
                 curr_brim_hidden_states), brim_hidden_state, \
                (curr_latent_sample, curr_latent_mean, curr_latent_logvar,
                 curr_task_inference_hidden_states), task_inference_hidden_state, curr_extra_information = self.brim(
                    curr_actions,
                    curr_states,
                    curr_rewards,
                    brim_hidden_state,
                    task_inference_hidden_state,
                    sample,
                    state_process=state_process)
                brim_output1.append(curr_output1)
                brim_output2.append(curr_output2)
                brim_output3.append(curr_output3)
                brim_output4.append(curr_output4)
                brim_output5.append(curr_output5)
                brim_hidden_states.append(curr_brim_hidden_states)
                if 'exploration_policy_embedded_state' in curr_extra_information:
                    if 'exploration_policy_embedded_state' not in extras_information:
                        extras_information['exploration_policy_embedded_state'] = []
                    extras_information['exploration_policy_embedded_state'].append(curr_extra_information['exploration_policy_embedded_state'])
                if 'exploitation_policy_embedded_state' in curr_extra_information:
                    if 'exploitation_policy_embedded_state' not in extras_information:
                        extras_information['exploitation_policy_embedded_state'] = []
                    extras_information['exploitation_policy_embedded_state'].append(curr_extra_information['exploitation_policy_embedded_state'])

                latent_sample.append(curr_latent_sample)
                latent_mean.append(curr_latent_mean)
                latent_logvar.append(curr_latent_logvar)
                task_inference_hidden_states.append(curr_task_inference_hidden_states)

                # select hx and ignore cx ( use GRU as RNN network)
                brim_hidden_state = brim_hidden_state[0].detach()

                task_inference_hidden_state = task_inference_hidden_state.detach()

                if self.use_stateful_vision_core and policy is not None:
                    state_encoder.detach_hidden_state()

            brim_output1 = torch.cat(brim_output1, dim=0)
            brim_output2 = torch.cat(brim_output2, dim=0)
            brim_output3 = torch.cat(brim_output3, dim=0)
            brim_output4 = torch.cat(brim_output4, dim=0)
            brim_output5 = torch.cat(brim_output5, dim=0)
            brim_hidden_states = torch.cat(brim_hidden_states, dim=0)
            if 'exploration_policy_embedded_state' in extras_information:
                extras_information['exploration_policy_embedded_state'] = torch.cat(extras_information['exploration_policy_embedded_state'], dim=0)
            if 'exploitation_policy_embedded_state' in extras_information:
                extras_information['exploitation_policy_embedded_state'] = torch.cat(extras_information['exploitation_policy_embedded_state'], dim=0)

            latent_sample = torch.cat(latent_sample, dim=0)
            latent_mean = torch.cat(latent_mean, dim=0)
            latent_logvar = torch.cat(latent_logvar, dim=0)
            task_inference_hidden_states = torch.cat(task_inference_hidden_states, dim=0)

        if return_prior:
            latent_sample = torch.cat((prior_sample, latent_sample))
            latent_mean = torch.cat((prior_mean, latent_mean))
            latent_logvar = torch.cat((prior_logvar, latent_logvar))
            task_inference_hidden_states = torch.cat((prior_task_inference_hidden_state, task_inference_hidden_states))

            brim_output1 = torch.cat((prior_brim_output1, brim_output1))
            brim_output2 = torch.cat((prior_brim_output2, brim_output2))
            brim_output3 = torch.cat((prior_brim_output3, brim_output3))
            brim_output4 = torch.cat((prior_brim_output4, brim_output4))
            brim_output5 = torch.cat((prior_brim_output5, brim_output5))
            brim_hidden_states = torch.cat((prior_brim_hidden_state, brim_hidden_states))

            if 'exploration_policy_embedded_state' in extras_information:
                extras_information['exploration_policy_embedded_state'] = torch.cat((prior_extras_information['exploration_policy_embedded_state'].unsqueeze(0), extras_information['exploration_policy_embedded_state']))
            if 'exploitation_policy_embedded_state' in extras_information:
                extras_information['exploitation_policy_embedded_state'] = torch.cat((prior_extras_information['exploitation_policy_embedded_state'].unsqueeze(0), extras_information['exploitation_policy_embedded_state']))

        if latent_mean.shape[0] == 1:
            latent_sample, latent_mean, latent_logvar = latent_sample[0], latent_mean[0], latent_logvar[0]

        if brim_output1.shape[0] == 1:
            brim_output1 = brim_output1[0]

        if brim_output2.shape[0] == 1:
            brim_output2 = brim_output2[0]

        if brim_output3.shape[0] == 1:
            brim_output3 = brim_output3[0]

        if brim_output4.shape[0] == 1:
            brim_output4 = brim_output4[0]

        if brim_output5.shape[0] == 1:
            brim_output5 = brim_output5[0]

        if 'exploration_policy_embedded_state' in extras_information:
            extras_information['exploration_policy_embedded_state'] = extras_information['exploration_policy_embedded_state'].squeeze(0)

        if 'exploitation_policy_embedded_state' in extras_information:
            extras_information['exploitation_policy_embedded_state'] = extras_information['exploitation_policy_embedded_state'].squeeze(0)

        return brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_states, \
               latent_sample, latent_mean, latent_logvar, task_inference_hidden_states, extras_information

    def forward_exploration_branch(self,
                                   actions,
                                   states,
                                   rewards,
                                   brim_hidden_state,
                                   task_inference_hidden_state,
                                   return_prior,
                                   sample,
                                   detach_every,
                                   policy,
                                   prev_state=None,
                                   rpe=None
                                   ):
        brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_states, \
        latent_sample, latent_mean, latent_logvar, task_inference_hidden_states, extras_information = self.forward(
            actions,
            states,
            rewards,
            brim_hidden_state,
            task_inference_hidden_state,
            return_prior,
            sample,
            detach_every,
            activated_branch='exploration',
            policy=policy,
            prev_state=prev_state,
            rpe=rpe)
        exploration_policy_embedded_state = None
        if 'exploration_policy_embedded_state' in extras_information:
            exploration_policy_embedded_state = extras_information['exploration_policy_embedded_state']

        return brim_output1, brim_output3, brim_output5, brim_hidden_states, \
               latent_sample, latent_mean, latent_logvar, task_inference_hidden_states, exploration_policy_embedded_state

    def forward_exploitation_branch(self,
                                    actions,
                                    states,
                                    rewards,
                                    brim_hidden_state,
                                    task_inference_hidden_state,
                                    return_prior,
                                    sample,
                                    detach_every,
                                    policy,
                                    prev_state=None,
                                    rpe=None
                                    ):
        brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_states, \
        latent_sample, latent_mean, latent_logvar, task_inference_hidden_states, extras_information = self.forward(
            actions,
            states,
            rewards,
            brim_hidden_state,
            task_inference_hidden_state,
            return_prior,
            sample,
            detach_every,
            activated_branch='exploitation',
            policy=policy,
            prev_state=prev_state,
            rpe=rpe)
        exploitation_policy_embedded_state = None
        if 'exploitation_policy_embedded_state' in extras_information:
            exploitation_policy_embedded_state = extras_information['exploitation_policy_embedded_state']

        return brim_output2, brim_output4, brim_output5, brim_hidden_states, \
               latent_sample, latent_mean, latent_logvar, task_inference_hidden_states, exploitation_policy_embedded_state

    def forward_level3(self,
                       actions,
                       states,
                       rewards,
                       brim_hidden_state,
                       task_inference_hidden_state,
                       return_prior,
                       sample,
                       detach_every,
                       prev_state
                       ):

        brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_states, \
        latent_sample, latent_mean, latent_logvar, task_inference_hidden_states, _ = self.forward(
            actions,
            states,
            rewards,
            brim_hidden_state,
            task_inference_hidden_state,
            return_prior,
            sample,
            detach_every,
            activated_branch='level3',
            policy=None,
            prev_state=prev_state,
            rpe=None)

        return brim_output5, latent_mean, latent_logvar
