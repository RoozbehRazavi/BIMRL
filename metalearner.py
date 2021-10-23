import os
import time

import gym
import numpy as np
import torch

from algorithms.online_storage import OnlineStorage
from algorithms.ppo import PPO
from environments.parallel_envs import make_vec_envs
from models.policy import Policy
from utils import evaluation as utl_eval
from utils import helpers as utl
from utils.tb_logger import TBLogger
from base2final import Base2Final
from utils.visualize import visualize_policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MetaLearner:
    """
    Meta-Learner class with the main training loop for variBAD.
    """

    def __init__(self, args):

        self.args = args
        utl.seed(self.args.seed, self.args.deterministic_execution)

        # calculate number of updates and keep count of frames/iterations
        self.num_updates = int(args.num_frames) // args.policy_num_steps // args.num_processes
        self.in_this_run_frames = 0
        self.total_frames = 0
        self.iter_idx = 0

        # initialise tensorboard logger
        self.logger = TBLogger(self.args, self.args.exp_label, self.args.env_name)

        self.exploration_num_processes = int(args.exploration_processes_portion * args.num_processes)
        self.exploitation_num_processes = int((1 - args.exploration_processes_portion) * args.num_processes)
        # initialise environments
        train_exploration = True
        train_exploitation = True
        if self.args.exploration_processes_portion == 0.0:
            train_exploration = False
        if self.args.exploration_processes_portion == 1.0:
            train_exploitation = False

        self.exploration_envs = None
        self.exploitation_envs = None
        if train_exploration:
            self.exploration_envs = make_vec_envs(env_name=args.env_name, seed=args.seed,
                                                  num_processes=self.exploration_num_processes,
                                                  gamma=args.policy_gamma, device=device,
                                                  episodes_per_task=self.args.max_rollouts_per_task,
                                                  normalise_rew=args.norm_rew_for_policy, ret_rms=None)
        if train_exploitation:
            self.exploitation_envs = make_vec_envs(env_name=args.env_name, seed=args.seed,
                                                   num_processes=self.exploitation_num_processes,
                                                   gamma=args.policy_gamma, device=device,
                                                   episodes_per_task=self.args.max_rollouts_per_task,
                                                   normalise_rew=args.norm_rew_for_policy, ret_rms=None)

        envs = self.exploration_envs if self.exploration_envs is not None else self.exploitation_envs
        # calculate what the maximum length of the trajectories is
        self.args.max_trajectory_len = envs._max_episode_steps
        self.args.max_trajectory_len *= self.args.max_rollouts_per_task

        # get policy input dimensions
        self.args.state_dim = envs.observation_space.shape[0]
        self.args.task_dim = envs.task_dim
        self.args.belief_dim = envs.belief_dim
        self.args.num_states = envs.num_states
        # get policy output (action) dimensions
        self.args.action_space = envs.action_space
        if isinstance(envs.action_space, gym.spaces.discrete.Discrete):
            self.args.action_dim = 1
        else:
            self.args.action_dim = envs.action_space.shape[0]

        # initialise VAE and policy
        self.base2final = Base2Final(self.args, self.logger, lambda: self.iter_idx, self.exploration_num_processes, self.exploitation_num_processes)
        self.exploration_policy_storage = self.initialise_policy_storage(self.exploration_num_processes)
        self.exploitation_policy_storage = self.initialise_policy_storage(self.exploitation_num_processes)
        self.exploration_policy = None
        self.exploitation_policy = None
        if train_exploration:
            self.exploration_policy = self.initialise_policy(policy_type='exploration')
        if train_exploitation:
            self.exploitation_policy = self.initialise_policy(policy_type='exploitation')

        self.state_prediction_running_normalizer = None
        self.action_prediction_running_normalizer = None
        self.reward_prediction_running_normalizer = None
        if train_exploration:
            self.state_prediction_running_normalizer = utl.RunningMeanStd(shape=(1,))
            self.action_prediction_running_normalizer = utl.RunningMeanStd(shape=(1,))
            self.reward_prediction_running_normalizer = utl.RunningMeanStd(shape=(1,))


        self.start_idx = 0
        if self.args.load_model and os.path.exists(os.path.join(self.logger.full_output_folder, 'models', 'brim_core.pt')):
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            general_info = torch.load(os.path.join(save_path, f"general.pt"), map_location=device)

            if self.base2final.state_decoder is not None:
                self.base2final.state_decoder.load_state_dict(torch.load(os.path.join(save_path, f"state_decoder.pt"), map_location=device))

            if self.base2final.reward_decoder is not None:
                self.base2final.reward_decoder.load_state_dict(
                    torch.load(os.path.join(save_path, f"reward_decoder.pt"), map_location=device))

            if self.base2final.action_decoder is not None:
                self.base2final.action_decoder.load_state_dict(
                    torch.load(os.path.join(save_path, f"action_decoder.pt"), map_location=device))

            if self.base2final.exploration_value_decoder is not None:
                self.base2final.exploration_value_decoder.load_state_dict(
                    torch.load(os.path.join(save_path, f"exploration_value_decoder.pt"), map_location=device))

            if self.base2final.exploitation_value_decoder is not None:
                self.base2final.exploitation_value_decoder.load_state_dict(
                    torch.load(os.path.join(save_path, f"exploitation_value_decoder.pt"), map_location=device))

            self.base2final.brim_core.load_state_dict(torch.load(os.path.join(save_path, f"brim_core.pt"), map_location=device))

            if self.exploration_policy is not None:
                self.exploration_policy.actor_critic.load_state_dict(
                    torch.load(os.path.join(save_path, f"exploration_policy.pt"), map_location=device))
            if self.exploitation_policy is not None:
                self.exploitation_policy.actor_critic.load_state_dict(
                    torch.load(os.path.join(save_path, f"exploitation_policy.pt"), map_location=device))

            self.start_idx = general_info['iter_idx']
            self.iter_idx = self.start_idx
            self.total_frames = self.start_idx * args.policy_num_steps * args.num_processes
            self.base2final.optimiser_vae.load_state_dict(general_info['vae_optimiser'])
            if self.exploration_policy is not None:
                self.exploration_policy.optimiser.load_state_dict(general_info['exploration_policy_optimiser'])
            if self.exploitation_policy is not None:
                self.exploitation_policy.optimiser.load_state_dict(general_info['exploitation_policy_optimiser'])

            # if self.args.norm_rew_for_policy:
            #     if self.exploration_envs is not None:
            #         self.exploration_envs.venv.ret_rms = torch.load(os.path.join(save_path, 'env_rew_rms_exploration.pkl'), map_location=device)
            #     if self.exploitation_envs is not None:
            #         self.exploitation_envs.venv.ret_rms = torch.load(os.path.join(save_path, 'env_rew_rms_exploitation.pkl'), map_location=device)
            # if self.args.norm_state_for_policy and self.args.pass_state_to_policy:
            #     if self.exploration_policy is not None:
            #         self.exploration_policy.actor_critic.state_rms = torch.load(os.path.join(save_path, 'policy_state_rms_exploration.pkl'), map_location=device)
            #     if self.exploitation_policy is not None:
            #         self.exploitation_policy.actor_critic.state_rms = torch.load(os.path.join(save_path, 'policy_state_rms_exploitation.pkl'), map_location=device)
            # if self.args.norm_task_inference_latent_for_policy and self.args.pass_task_inference_latent_to_policy:
            #     if self.exploration_policy is not None:
            #         self.exploration_policy.actor_critic.task_inference_latent_rms = torch.load(os.path.join(save_path, 'policy_latent_rms_exploration.pkl'), map_location=device)
            #     if self.exploitation_policy is not None:
            #         self.exploitation_policy.actor_critic.task_inference_latent_rms = torch.load(os.path.join(save_path, 'policy_latent_rms_exploitation.pkl'), map_location=device)
            # if self.args.norm_rim_level1_output and self.args.use_rim_level1:
            #     if self.exploration_policy is not None:
            #         self.exploration_policy.actor_critic.rim_level1_output_rms = torch.load(os.path.join(save_path, 'policy_rim_level1_rms_exploration.pkl'), map_location=device)
            #     if self.exploitation_policy is not None:
            #         self.exploitation_policy.actor_critic.rim_level1_output_rms = torch.load(os.path.join(save_path, 'policy_rim_level1_rms_exploitation.pkl'), map_location=device)
            # if self.state_prediction_running_normalizer is not None:
            #     self.state_prediction_running_normalizer = torch.load(os.path.join(save_path, 'state_error_rms.pkl'), map_location=device)
            # if self.action_prediction_running_normalizer is not None:
            #     self.action_prediction_running_normalizer = torch.load(os.path.join(save_path, 'action_error_rms.pkl'), map_location=device)

    def initialise_policy_storage(self, num_processes):
        return OnlineStorage(args=self.args,
                             num_steps=self.args.policy_num_steps,
                             num_processes=num_processes,
                             state_dim=self.args.state_dim,
                             task_inference_latent_dim=self.args.task_inference_latent_dim,
                             belief_dim=self.args.belief_dim,
                             task_dim=self.args.task_dim,
                             action_space=self.args.action_space,
                             task_inference_hidden_size=self.args.vae_encoder_gru_hidden_size,
                             brim_hidden_size=max(self.args.rim_level1_hidden_size, self.args.rim_level2_hidden_size,
                                                  self.args.rim_level3_hidden_size),
                             normalise_rewards=self.args.norm_rew_for_policy,
                             )

    def initialise_policy(self, policy_type):
        if policy_type == 'exploration':
            envs = self.exploration_envs
        elif policy_type == 'exploitation':
            envs = self.exploitation_envs
        else:
            raise NotImplementedError

        if hasattr(envs.action_space, 'low'):
            action_low = envs.action_space.low
            action_high = envs.action_space.high
        else:
            action_low = action_high = None

        # initialise policy network
        policy_net = Policy(
            args=self.args,
            #
            pass_state_to_policy=self.args.pass_state_to_policy,
            pass_task_inference_latent_to_policy=self.args.pass_task_inference_latent_to_policy,
            pass_belief_to_policy=self.args.pass_belief_to_policy,
            pass_task_to_policy=self.args.pass_task_to_policy,
            pass_rim_level1_output_to_policy=self.args.use_rim_level1,
            dim_state=self.args.state_dim,
            task_inference_latent_dim=self.args.task_inference_latent_dim,
            rim_level1_output_dim=self.args.rim_level1_output_dim,
            dim_belief=self.args.belief_dim,
            dim_task=self.args.task_dim,
            #
            hidden_layers=self.args.policy_layers,
            activation_function=self.args.policy_activation_function,
            policy_initialisation=self.args.policy_initialisation,
            #
            action_space=envs.action_space,
            init_std=self.args.policy_init_std,
            norm_actions_of_policy=self.args.norm_actions_of_policy,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        if self.args.policy == 'ppo':
            policy = PPO(
                self.args,
                policy_net,
                self.args.policy_value_loss_coef,
                self.args.policy_entropy_coef,
                policy_optimiser=self.args.policy_optimiser,
                policy_anneal_lr=self.args.policy_anneal_lr,
                train_steps=self.num_updates,
                lr=self.args.lr_policy,
                eps=self.args.policy_eps,
                ppo_epoch=self.args.ppo_num_epochs,
                num_mini_batch=self.args.ppo_num_minibatch,
                use_huber_loss=self.args.ppo_use_huberloss,
                use_clipped_value_loss=self.args.ppo_use_clipped_value_loss,
                clip_param=self.args.ppo_clip_param,
                optimiser_vae=self.base2final.optimiser_vae,
            )
        else:
            raise NotImplementedError

        return policy

    def train(self):
        """ Main Meta-Training loop """
        start_time = time.time()

        train_exploration = True
        train_exploitation = True
        if self.args.exploration_processes_portion == 0.0:
            train_exploration = False
        if self.args.exploration_processes_portion == 1.0:
            train_exploitation = False

        # reset environments
        if train_exploration:
            exploration_prev_state, exploration_belief, exploration_task = utl.reset_env(self.exploration_envs,
                                                                                         self.args)
        if train_exploitation:
            exploitation_prev_state, exploitation_belief, exploitation_task = utl.reset_env(self.exploitation_envs,
                                                                                            self.args)

        # insert initial observation / embeddings to rollout storage

        if train_exploration:
            self.exploration_policy_storage.prev_state[0].copy_(exploration_prev_state)
        if train_exploitation:
            self.exploitation_policy_storage.prev_state[0].copy_(exploitation_prev_state)

        vae_is_pretrained = False
        for self.iter_idx in range(self.start_idx, self.num_updates):

            # First, re-compute the hidden states given the current rollouts (since the VAE might've changed)
            with torch.no_grad():
                if train_exploration:
                    brim_output1, brim_output3, exploration_brim_output5, exploration_brim_hidden_state,\
                    exploration_latent_sample, exploration_latent_mean, exploration_latent_logvar, exploration_task_inference_hidden_state,\
                        exploration_policy_embedded_state = self.encode_running_trajectory(
                        self.base2final.exploration_rollout_storage, activated_branch='exploration')

                if train_exploitation:
                    brim_output2, brim_output4, exploitation_brim_output5, exploitation_brim_hidden_state, \
                    exploitation_latent_sample, exploitation_latent_mean, exploitation_latent_logvar, exploitation_task_inference_hidden_state,\
                        exploitation_policy_embedded_state = self.encode_running_trajectory(
                        self.base2final.exploitation_rollout_storage, activated_branch='exploitation')

            # add this initial hidden state to the policy storage
            if train_exploration:
                assert len(self.exploration_policy_storage.latent_mean) == 0  # make sure we emptied buffers
                self.exploration_policy_storage.task_inference_hidden_states[0].copy_(
                    exploration_task_inference_hidden_state)
                self.exploration_policy_storage.latent_samples.append(exploration_latent_sample.clone())
                self.exploration_policy_storage.latent_mean.append(exploration_latent_mean.clone())
                self.exploration_policy_storage.latent_logvar.append(exploration_latent_logvar.clone())

                self.exploration_policy_storage.brim_hidden_states[0].copy_(exploration_brim_hidden_state)
                self.exploration_policy_storage.brim_output_level1.append(brim_output1)
                self.exploration_policy_storage.brim_output_level2.append(brim_output3)
                self.exploration_policy_storage.brim_output_level3.append(exploration_brim_output5)
                self.exploration_policy_storage.policy_embedded_state.append(exploration_policy_embedded_state)
                state_errors = []
                action_errors = []
                reward_errors = []

            if train_exploitation:
                if hasattr(self.exploitation_policy_storage, 'latent_mean'):
                    assert len(self.exploitation_policy_storage.latent_mean) == 0  # make sure we emptied buffers
                elif hasattr(self.exploitation_policy_storage, 'brim_output_level1'):
                    assert len(self.exploitation_policy_storage.brim_output_level1) == 0  # make sure we emptied buffers
                else:
                    print('Policy independent on indeed task')
                self.exploitation_policy_storage.task_inference_hidden_states[0].copy_(
                    exploitation_task_inference_hidden_state)
                self.exploitation_policy_storage.latent_samples.append(exploitation_latent_sample.clone())
                self.exploitation_policy_storage.latent_mean.append(exploitation_latent_mean.clone())
                self.exploitation_policy_storage.latent_logvar.append(exploitation_latent_logvar.clone())

                self.exploitation_policy_storage.brim_hidden_states[0].copy_(exploitation_brim_hidden_state)
                self.exploitation_policy_storage.brim_output_level1.append(brim_output2)
                self.exploitation_policy_storage.brim_output_level2.append(brim_output4)
                self.exploitation_policy_storage.brim_output_level3.append(exploitation_brim_output5)
                self.exploitation_policy_storage.policy_embedded_state.append(exploitation_policy_embedded_state)

            # rollout policies for a few steps
            for step in range(self.args.policy_num_steps):

                # sample actions from policy
                with torch.no_grad():
                    if train_exploration:
                        exploration_value, exploration_action, exploration_action_log_prob = utl.select_action(
                            args=self.args,
                            policy=self.exploration_policy,
                            belief=exploration_belief,
                            task=exploration_task,
                            deterministic=False,
                            latent_sample=exploration_latent_sample,
                            latent_mean=exploration_latent_mean,
                            latent_logvar=exploration_latent_logvar,
                            brim_output_level1=brim_output1,
                            policy_embedded_state=exploration_policy_embedded_state
                        )
                    if train_exploitation:
                        exploitation_value, exploitation_action, exploitation_action_log_prob = utl.select_action(
                            args=self.args,
                            policy=self.exploitation_policy,
                            belief=exploitation_belief,
                            task=exploitation_task,
                            deterministic=False,
                            latent_sample=exploitation_latent_sample,
                            latent_mean=exploitation_latent_mean,
                            latent_logvar=exploitation_latent_logvar,
                            brim_output_level1=brim_output2,
                            policy_embedded_state=exploitation_policy_embedded_state,
                        )

                # take step in the environment
                if train_exploration:
                    [exploration_next_state, exploration_belief, exploration_task], \
                    (exploration_rew_raw, exploration_rew_normalised), \
                    exploration_done, exploration_infos = utl.env_step(self.exploration_envs, exploration_action, self.args)

                    latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                       add_nonlinearity_to_latent=self.args.add_nonlinearity_to_latent,
                                                       latent_sample=exploration_latent_sample,
                                                       latent_mean=exploration_latent_mean,
                                                       latent_logvar=exploration_latent_logvar)
                    if self.args.use_rim_level3:
                        if self.args.residual_task_inference_latent:
                            latent = torch.cat((exploration_brim_output5.squeeze(0), latent), dim=-1)
                        else:
                            latent = exploration_brim_output5

                    exploration_intrinsic_rew_raw, \
                    exploration_intrinsic_rew_normalised, state_error, action_error, reward_error = utl.compute_intrinsic_reward(
                        exploration_rew_raw,
                        exploration_rew_normalised,
                        latent=latent,
                        prev_state=exploration_prev_state,
                        next_state=exploration_next_state,
                        action=exploration_action.float(),
                        decode_action=self.args.decode_action,
                        state_decoder=self.base2final.state_decoder,
                        action_decoder=self.base2final.action_decoder,
                        state_prediction_running_normalizer=self.state_prediction_running_normalizer,
                        action_prediction_running_normalizer=self.action_prediction_running_normalizer,
                        state_prediction_intrinsic_reward_coef=self.args.state_prediction_intrinsic_reward_coef,
                        action_prediction_intrinsic_reward_coef=self.args.action_prediction_intrinsic_reward_coef,
                        extrinsic_reward_intrinsic_reward_coef=self.args.extrinsic_reward_intrinsic_reward_coef,
                        reward_decoder=self.base2final.reward_decoder,
                        reward_prediction_intrinsic_reward_coef=self.args.reward_prediction_intrinsic_reward_coef,
                        decode_reward=self.args.decode_reward,
                        reward_prediction_running_normalizer=self.reward_prediction_running_normalizer,
                        rew_pred_type=self.args.rew_pred_type,
                        itr_idx=self.iter_idx,
                        num_updates=self.num_updates
                        )
                    state_errors.append(state_error)
                    action_errors.append(action_error)
                    reward_errors.append(reward_error)

                    exploration_done_episode = list()
                    for i in range(self.exploration_num_processes):
                        exploration_done_episode.append(1.0 if exploration_infos[i]['done_mdp'] else 0.0)
                    exploration_done_episode = torch.Tensor(exploration_done_episode).float().to(device).unsqueeze(1)

                    exploration_done = torch.from_numpy(np.array(exploration_done, dtype=int)).to(device).float().view((-1, 1))
                    # create mask for episode ends
                    exploration_masks_done = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in exploration_done]).to(device)
                    # bad_mask is true if episode ended because time limit was reached
                    exploration_bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in exploration_infos]).to(device)

                if train_exploitation:
                    [exploitation_next_state, exploitation_belief, exploitation_task], \
                    (exploitation_rew_raw, exploitation_rew_normalised), \
                    exploitation_done, exploitation_infos = utl.env_step(self.exploitation_envs, exploitation_action, self.args)

                    exploitation_done_episode = list()
                    for i in range(self.exploitation_num_processes):
                        exploitation_done_episode.append(1.0 if exploitation_infos[i]['done_mdp'] else 0.0)
                    exploitation_done_episode = torch.Tensor(exploitation_done_episode).float().to(device).unsqueeze(1)

                    exploitation_done = torch.from_numpy(np.array(exploitation_done, dtype=int)).to(device).float().view((-1, 1))
                    # create mask for episode ends
                    exploitation_masks_done = torch.FloatTensor(
                        [[0.0] if done_ else [1.0] for done_ in exploitation_done]).to(device)
                    # bad_mask is true if episode ended because time limit was reached
                    exploitation_bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in exploitation_infos]).to(device)

                with torch.no_grad():
                    # compute next embedding (for next loop and/or value prediction bootstrap)
                    if train_exploration:
                        # compute RPE
                        if self.args.use_memory and self.args.use_rpe and self.args.decode_reward:
                            reward_decoder = self.base2final.reward_decoder()
                            latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                               add_nonlinearity_to_latent=self.args.add_nonlinearity_to_latent,
                                                               latent_sample=exploration_latent_sample,
                                                               latent_mean=exploration_latent_mean,
                                                               latent_logvar=exploration_latent_logvar)
                            if self.args.use_rim_level3:
                                if self.args.residual_task_inference_latent:
                                    latent = torch.cat((exploration_brim_output5.squeeze(0), latent), dim=-1)
                                else:
                                    latent = exploration_brim_output5

                            rpe = exploration_rew_raw - reward_decoder(reward_decoder, latent, exploration_next_state, prev_state=exploration_prev_state, action=exploration_action, n_step_reward_prediction=False)
                        else:
                            rpe = 0.1 * torch.ones(size=(self.exploration_num_processes, 1))
                        brim_output1, brim_output3, brim_output5, exploration_brim_hidden_state, exploration_latent_sample, exploration_latent_mean, exploration_latent_logvar, \
                        exploration_task_inference_hidden_state, exploration_policy_embedded_state = utl.update_encoding(
                            policy=self.exploration_policy.actor_critic,
                            brim_core=self.base2final.brim_core,
                            next_obs=exploration_next_state,
                            action=exploration_action,
                            reward=exploration_intrinsic_rew_raw,
                            done=exploration_done,
                            task_inference_hidden_state=exploration_task_inference_hidden_state,
                            brim_hidden_state=exploration_brim_hidden_state,
                            activated_branch='exploration',
                            done_episode=exploration_done_episode,
                            rpe=rpe)
                    if train_exploitation:
                        # compute RPE
                        if self.args.use_memory and self.args.use_rpe and self.args.decode_reward:
                            reward_decoder = self.base2final.reward_decoder()
                            latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                               add_nonlinearity_to_latent=self.args.add_nonlinearity_to_latent,
                                                               latent_sample=exploitation_latent_sample,
                                                               latent_mean=exploitation_latent_mean,
                                                               latent_logvar=exploitation_latent_logvar)
                            if self.args.use_rim_level3:
                                if self.args.residual_task_inference_latent:
                                    latent = torch.cat((exploitation_brim_output5.squeeze(0), latent), dim=-1)
                                else:
                                    latent = exploitation_brim_output5

                            rpe = exploitation_rew_raw - reward_decoder(reward_decoder, latent, exploitation_next_state,
                                                                       prev_state=exploitation_prev_state,
                                                                       action=exploitation_action,
                                                                       n_step_reward_prediction=False)
                        else:
                            rpe = 0.1 * torch.ones(size=(self.exploitation_num_processes, 1))
                        brim_output2, brim_output4, brim_output5, exploitation_brim_hidden_state, exploitation_latent_sample, exploitation_latent_mean, exploitation_latent_logvar, \
                        exploitation_task_inference_hidden_state, exploitation_policy_embedded_state = utl.update_encoding(
                            brim_core=self.base2final.brim_core,
                            policy=self.exploitation_policy.actor_critic,
                            next_obs=exploitation_next_state,
                            action=exploitation_action,
                            reward=exploitation_rew_raw,
                            done=exploitation_done,
                            task_inference_hidden_state=exploitation_task_inference_hidden_state,
                            brim_hidden_state=exploitation_brim_hidden_state,
                            activated_branch='exploitation',
                            done_episode=exploitation_done_episode,
                            rpe=rpe)

                # before resetting, update the embedding and add to vae buffer
                # (last state might include useful task info)
                if not (self.args.disable_decoder and self.args.disable_stochasticity_in_latent):
                    if train_exploration:

                        self.base2final.exploration_rollout_storage.insert(exploration_prev_state.clone(),
                                                                           exploration_action.detach().clone(),
                                                                           exploration_next_state.clone(),
                                                                           exploration_rew_raw.clone(),
                                                                           exploration_done.clone(),
                                                                           exploration_task.clone() if exploration_task is not None else None,
                                                                           exploration_masks_done,
                                                                           exploration_bad_masks,
                                                                           intrinsic_rewards=exploration_intrinsic_rew_normalised if self.args.norm_rew_for_policy else exploration_intrinsic_rew_raw,
                                                                           done_task=exploration_done.clone(),
                                                                           done_episode=exploration_done_episode.clone())
                    if train_exploitation:
                        self.base2final.exploitation_rollout_storage.insert(exploitation_prev_state.clone(),
                                                                            exploitation_action.detach().clone(),
                                                                            exploitation_next_state.clone(),
                                                                            exploitation_rew_raw.clone(),
                                                                            exploitation_done.clone(),
                                                                            exploitation_task.clone() if exploitation_task is not None else None,
                                                                            exploitation_masks_done,
                                                                            exploitation_bad_masks,
                                                                            intrinsic_rewards=None,
                                                                            done_task=exploitation_done.clone(),
                                                                            done_episode=exploitation_done_episode)

                if self.args.rlloss_through_encoder:
                    # add the obs before reset to the policy storage
                    # (only used to recompute embeddings if rlloss is backpropagated through encoder)
                    if train_exploration:
                        self.exploration_policy_storage.next_state[step] = exploration_next_state.clone()

                    if train_exploitation:
                        self.exploitation_policy_storage.next_state[step] = exploitation_next_state.clone()

                # reset environments that are done
                if train_exploration:
                    done_indices = np.argwhere(exploration_done.cpu().flatten()).flatten()
                    if len(done_indices) > 0:
                        exploration_next_state, exploration_belief, exploration_task = utl.reset_env(
                            self.exploration_envs,
                            self.args,
                            indices=done_indices,
                            state=exploration_next_state)
                if train_exploitation:
                    done_indices = np.argwhere(exploitation_done.cpu().flatten()).flatten()
                    if len(done_indices) > 0:
                        exploitation_next_state, exploitation_belief, exploitation_task = utl.reset_env(
                            self.exploitation_envs,
                            self.args,
                            indices=done_indices,
                            state=exploitation_next_state)
                # add experience to policy buffer
                if train_exploration:
                    self.exploration_policy_storage.insert(
                        state=exploration_next_state,
                        belief=exploration_belief,
                        task=exploration_task,
                        actions=exploration_action,
                        action_log_probs=exploration_action_log_prob,
                        rewards_raw=exploration_intrinsic_rew_raw,
                        rewards_normalised=exploration_intrinsic_rew_normalised,
                        value_preds=exploration_value,
                        masks=exploration_masks_done,
                        bad_masks=exploration_bad_masks,
                        done=exploration_done,
                        done_episode=exploration_done_episode,
                        task_inference_hidden_states=exploration_task_inference_hidden_state.squeeze(0),
                        latent_sample=exploration_latent_sample,
                        latent_mean=exploration_latent_mean,
                        latent_logvar=exploration_latent_logvar,
                        brim_output_level1=brim_output1,
                        brim_output_level2=brim_output3,
                        brim_output_level3=exploration_brim_output5,
                        policy_embedded_state=exploration_policy_embedded_state,
                        brim_hidden_states=exploration_brim_hidden_state.squeeze(0)
                    )
                    exploration_prev_state = exploration_next_state
                if train_exploitation:
                    self.exploitation_policy_storage.insert(
                        state=exploitation_next_state,
                        belief=exploitation_belief,
                        task=exploitation_task,
                        actions=exploitation_action,
                        action_log_probs=exploitation_action_log_prob,
                        rewards_raw=exploitation_rew_raw,
                        rewards_normalised=exploitation_rew_normalised,
                        value_preds=exploitation_value,
                        masks=exploitation_masks_done,
                        bad_masks=exploitation_bad_masks,
                        done=exploitation_done,
                        done_episode=exploitation_done_episode,
                        task_inference_hidden_states=exploitation_task_inference_hidden_state.squeeze(0),
                        latent_sample=exploitation_latent_sample,
                        latent_mean=exploitation_latent_mean,
                        latent_logvar=exploitation_latent_logvar,
                        brim_output_level1=brim_output2,
                        brim_output_level2=brim_output4,
                        brim_output_level3=exploitation_brim_output5,
                        policy_embedded_state=exploitation_policy_embedded_state,
                        brim_hidden_states=exploitation_brim_hidden_state.squeeze(0)
                    )
                    exploitation_prev_state = exploitation_next_state

                self.total_frames += self.args.num_processes
                self.in_this_run_frames += self.args.num_processes

            # --- UPDATE ---

            if self.args.precollect_len <= self.in_this_run_frames:

                # check if we are pre-training the VAE
                if self.args.pretrain_len > 0 and not vae_is_pretrained:
                    for _ in range(self.args.pretrain_len):
                        self.base2final.compute_vae_loss(update=True)
                    vae_is_pretrained = True

                # otherwise do the normal update (policy + vae)
                else:
                    if train_exploration:
                        exploration_train_stats = self.update(
                            belief=exploration_belief,
                            task=exploration_task,
                            latent_sample=exploration_latent_sample,
                            latent_mean=exploration_latent_mean,
                            latent_logvar=exploration_latent_logvar,
                            brim_output_level1=brim_output1,
                            policy_embedded_state=exploration_policy_embedded_state,
                            policy=self.exploration_policy,
                            policy_storage=self.exploration_policy_storage,
                            activated_branch='exploration')
                    if train_exploitation:
                        #with torch.autograd.set_detect_anomaly(True):
                            exploitation_train_stats = self.update(
                                belief=exploitation_belief,
                                task=exploitation_task,
                                latent_sample=exploitation_latent_sample,
                                latent_mean=exploitation_latent_mean,
                                latent_logvar=exploitation_latent_logvar,
                                brim_output_level1=brim_output2,
                                policy_embedded_state=exploitation_policy_embedded_state,
                                policy=self.exploitation_policy,
                                policy_storage=self.exploitation_policy_storage,
                                activated_branch='exploitation')

                    # log
                    with torch.no_grad():
                        if train_exploration:
                            exploration_run_stats = [exploration_action, exploration_action_log_prob, exploration_value]
                            self.log(exploration_run_stats,
                                     exploration_train_stats,
                                     start_time,
                                     policy=self.exploration_policy,
                                     policy_storage=self.exploration_policy_storage,
                                     envs=self.exploration_envs,
                                     policy_type='exploration',
                                     meta_eval=train_exploration and train_exploitation,
                                     tmp=train_exploration and not train_exploitation
                                     )
                        if train_exploitation:
                            exploitation_run_stats = [exploitation_action, exploitation_action_log_prob,
                                                      exploitation_value]
                            self.log(exploitation_run_stats,
                                     exploitation_train_stats,
                                     start_time,
                                     policy=self.exploitation_policy,
                                     policy_storage=self.exploitation_policy_storage,
                                     envs=self.exploitation_envs,
                                     policy_type='exploitation',
                                     meta_eval=train_exploration and train_exploitation,
                                     tmp=train_exploration and not train_exploitation
                                     )

            # clean up after update
            if train_exploration:
                # TODO cat action_errors and state_errors
                self.exploration_policy_storage.after_update()
                self.state_prediction_running_normalizer.update(torch.cat(state_errors))
                self.action_prediction_running_normalizer.update(torch.cat(action_errors))
                self.reward_prediction_running_normalizer.update(torch.cat(reward_errors))
                state_errors = []
                action_errors = []
                reward_errors = []
            if train_exploitation:
                self.exploitation_policy_storage.after_update()

    def encode_running_trajectory(self, rollout_storage, activated_branch):
        """
        (Re-)Encodes (for each process) the entire current trajectory.
        Returns sample/mean/logvar and hidden state (if applicable) for the current timestep.
        :return:
        """

        # for each process, get the current batch (zero-padded obs/act/rew + length indicators)
        prev_obs, next_obs, act, rew, lens = rollout_storage.get_running_batch()

        # get embedding - will return (1+sequence_len) * batch * input_size -- includes the prior!
        if activated_branch == 'exploration':
            all_brim_output1, all_brim_output3, all_brim_output5, all_brim_hidden_states, \
            all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states, all_exploration_policy_embedded_state = self.base2final.brim_core.forward_exploration_branch(
                actions=act,
                states=next_obs,
                rewards=rew,
                brim_hidden_state=None,
                task_inference_hidden_state=None,
                return_prior=True,
                sample=True,
                detach_every=None,
                policy=self.exploration_policy.actor_critic,
                prev_state=prev_obs[0, :, :])
            # get the embedding / hidden state of the current time step (need to do this since we zero-padded)
            latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
            latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
            latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
            task_inference_hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(
                device)
            brim_output1 = (torch.stack([all_brim_output1[lens[i]][i] for i in range(len(lens))])).to(device)
            exploration_policy_embedded_state = (torch.stack([all_exploration_policy_embedded_state[lens[i]][i] for i in range(len(lens))])).to(device)
            brim_output3 = (torch.stack([all_brim_output3[lens[i]][i] for i in range(len(lens))])).to(device)
            brim_output5 = (torch.stack([all_brim_output5[lens[i]][i] for i in range(len(lens))])).to(device)
            brim_hidden_state = (torch.stack([all_brim_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)
            return brim_output1, brim_output3, brim_output5, brim_hidden_state, latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, exploration_policy_embedded_state
        elif activated_branch == 'exploitation':
            all_brim_output2, all_brim_output4, all_brim_output5, all_brim_hidden_states, \
            all_latent_samples, all_latent_means, all_latent_logvars, all_hidden_states, all_exploitation_policy_embedded_state = self.base2final.brim_core.forward_exploitation_branch(
                actions=act,
                states=next_obs,
                rewards=rew,
                brim_hidden_state=None,
                task_inference_hidden_state=None,
                return_prior=True,
                sample=True,
                detach_every=None,
                policy=self.exploitation_policy.actor_critic,
                prev_state=prev_obs[0, :, :])
            latent_sample = (torch.stack([all_latent_samples[lens[i]][i] for i in range(len(lens))])).to(device)
            latent_mean = (torch.stack([all_latent_means[lens[i]][i] for i in range(len(lens))])).to(device)
            latent_logvar = (torch.stack([all_latent_logvars[lens[i]][i] for i in range(len(lens))])).to(device)
            task_inference_hidden_state = (torch.stack([all_hidden_states[lens[i]][i] for i in range(len(lens))])).to(
                device)
            brim_output2 = (torch.stack([all_brim_output2[lens[i]][i] for i in range(len(lens))])).to(device)
            brim_output4 = (torch.stack([all_brim_output4[lens[i]][i] for i in range(len(lens))])).to(device)
            brim_output5 = (torch.stack([all_brim_output5[lens[i]][i] for i in range(len(lens))])).to(device)
            exploitation_policy_embedded_state = (
                torch.stack([all_exploitation_policy_embedded_state[lens[i]][i] for i in range(len(lens))])).to(device)
            brim_hidden_state = (torch.stack([all_brim_hidden_states[lens[i]][i] for i in range(len(lens))])).to(device)
            return brim_output2, brim_output4, brim_output5, brim_hidden_state, latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, exploitation_policy_embedded_state
        else:
            raise NotImplementedError

    def get_value(self, embedded_state, belief, task, latent_sample, latent_mean, latent_logvar, brim_output_level1, policy):
        latent = utl.get_latent_for_policy(sample_embeddings=self.args.sample_embeddings,
                                           add_nonlinearity_to_latent=self.args.add_nonlinearity_to_latent,
                                           latent_sample=latent_sample, latent_mean=latent_mean,
                                           latent_logvar=latent_logvar)
        return policy.actor_critic.get_value(embedded_state=embedded_state, belief=belief, task=task, latent=latent,
                                             brim_output_level1=brim_output_level1).detach()

    def update(self, policy_embedded_state, belief, task, latent_sample, latent_mean, latent_logvar, brim_output_level1, policy,
               policy_storage, activated_branch):
        """
        Meta-update.
        Here the policy is updated for good average performance across tasks.
        :return:
        """
        # bootstrap next value prediction
        with torch.no_grad():
            next_value = self.get_value(embedded_state=policy_embedded_state,
                                        belief=belief,
                                        task=task,
                                        latent_sample=latent_sample,
                                        latent_mean=latent_mean,
                                        latent_logvar=latent_logvar,
                                        brim_output_level1=brim_output_level1,
                                        policy=policy)

        # compute returns for current rollouts
        policy_storage.compute_returns(next_value, self.args.policy_use_gae, self.args.policy_gamma,
                                       self.args.policy_tau,
                                       use_proper_time_limits=self.args.use_proper_time_limits)

        # update agent (this will also call the VAE update!)
        policy_train_stats = policy.update(
            policy_storage=policy_storage,
            encoder=self.base2final.brim_core,
            rlloss_through_encoder=self.args.rlloss_through_encoder,
            compute_vae_loss=self.base2final.compute_vae_loss,
            compute_n_step_value_prediction_loss=self.base2final.compute_n_step_value_prediction_loss,
            compute_memory_loss=self.base2final.compute_memory_loss,
            activated_branch=activated_branch)

        return policy_train_stats

    def log(self, run_stats, train_stats, start_time, policy, policy_storage, envs, policy_type, meta_eval, tmp):

        if (self.iter_idx % self.args.meta_evaluate_interval == 0) and meta_eval and policy_type=='exploitation':
            utl_eval.evaluate_meta_policy(
                self.args,
                self.exploration_policy,
                self.exploitation_policy,
                envs.venv.ret_rms,
                self.iter_idx,
                self.base2final.state_decoder,
                self.base2final.action_decoder,
                self.state_prediction_running_normalizer,
                self.action_prediction_running_normalizer,
                self.reward_prediction_running_normalizer,
                self.base2final.brim_core,
                self.args.exploration_num_episodes,
                save_path=self.logger.full_output_folder)

        # --- visualize policy ----
        if self.iter_idx % self.args.vis_interval == self.args.vis_interval-1 and not policy_type == 'meta_policy':
            print('visualize ...')
            ret_rms = envs.venv.ret_rms if self.args.norm_rew_for_policy else None
            utl_eval.visualize_policy(
                args=self.args,
                policy=policy,
                ret_rms=ret_rms,
                brim_core=self.base2final.brim_core,
                iter_idx=self.iter_idx,
                policy_type=policy_type,
                state_decoder=self.base2final.state_decoder,
                action_decoder=self.base2final.action_decoder,
                num_episodes=1,
                state_prediction_running_normalizer=self.state_prediction_running_normalizer,
                action_prediction_running_normalizer=self.action_prediction_running_normalizer,
                reward_prediction_running_normalizer=self.reward_prediction_running_normalizer,
                full_output_folder=self.logger.full_output_folder,
                reward_decoder=self.base2final.reward_decoder,
                num_updates=self.num_updates)

        # --- evaluate policy ----

        if self.iter_idx % self.args.eval_interval == 0:
            print('evaluate ...')
            ret_rms = envs.venv.ret_rms if self.args.norm_rew_for_policy else None

            returns_per_episode, returns_per_episode__ = utl_eval.evaluate(
                args=self.args,
                policy=policy,
                ret_rms=ret_rms,
                brim_core=self.base2final.brim_core,
                iter_idx=self.iter_idx,
                policy_type=policy_type,
                state_decoder=self.base2final.state_decoder,
                action_decoder=self.base2final.action_decoder,
                state_prediction_running_normalizer=self.state_prediction_running_normalizer,
                action_prediction_running_normalizer=self.action_prediction_running_normalizer,
                reward_decoder=self.base2final.reward_decoder,
                reward_prediction_running_normalizer=self.reward_prediction_running_normalizer,
                tmp=tmp,
                num_updates=self.num_updates
                )

            # log the return avg/std across tasks (=processes)
            returns_avg = returns_per_episode.mean(dim=0)
            returns_std = returns_per_episode.std(dim=0)
            for k in range(len(returns_avg)):
                self.logger.add('return_avg_per_iter_{}/episode_{}'.format(policy_type, k + 1), returns_avg[k],
                                self.iter_idx)
                self.logger.add('return_avg_per_frame_{}/episode_{}'.format(policy_type, k + 1), returns_avg[k],
                                self.total_frames)
                self.logger.add('return_std_per_iter_{}/episode_{}'.format(policy_type, k + 1), returns_std[k],
                                self.iter_idx)
                self.logger.add('return_std_per_frame_{}/episode_{}'.format(policy_type, k + 1), returns_std[k],
                                self.total_frames)
            # print FPS only once
            print(f"Updates {self.iter_idx}, "
                  f"Frames {self.total_frames}, "
                  f"FPS {int(self.in_this_run_frames / (time.time() - start_time))}, "
                  f"\n Mean return {policy_type} (train): {returns_avg[-1].item()} \n")
            if tmp:
                returns_avg = returns_per_episode__.mean(dim=0)
                returns_std = returns_per_episode__.std(dim=0)
                for k in range(len(returns_avg)):
                    self.logger.add('return_avg_per_iter_{}/episode_{}'.format('exploitation', k + 1), returns_avg[k],
                                    self.iter_idx)
                    self.logger.add('return_avg_per_frame_{}/episode_{}'.format('exploitation', k + 1), returns_avg[k],
                                    self.total_frames)
                    self.logger.add('return_std_per_iter_{}/episode_{}'.format('exploitation', k + 1), returns_std[k],
                                    self.iter_idx)
                    self.logger.add('return_std_per_frame_{}/episode_{}'.format('exploitation', k + 1), returns_std[k],
                                    self.total_frames)
                # print FPS only once
                print(f"Updates {self.iter_idx}, "
                      f"Frames {self.total_frames}, "
                      f"\n Mean return exploitation (train): {returns_avg[-1].item()} \n")

        # --- save models ---
        if self.iter_idx % self.args.save_interval == 0:
            print('save ...')
            save_path = os.path.join(self.logger.full_output_folder, 'models')
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            idx_labels = ['']
            if self.args.save_intermediate_models:
                idx_labels.append(int(self.iter_idx))

            for idx_label in idx_labels:

                torch.save(policy.actor_critic.state_dict(), os.path.join(save_path, f"{policy_type}_policy{idx_label}.pt"),
                           _use_new_zipfile_serialization=False)
                torch.save(self.base2final.brim_core.state_dict(), os.path.join(save_path, f"brim_core{idx_label}.pt"),
                           _use_new_zipfile_serialization=False)
                if self.base2final.state_decoder is not None:
                    torch.save(self.base2final.state_decoder.state_dict(), os.path.join(save_path, f"state_decoder{idx_label}.pt"),
                               _use_new_zipfile_serialization=False)
                if self.base2final.reward_decoder is not None:
                    torch.save(self.base2final.reward_decoder.state_dict(), os.path.join(save_path, f"reward_decoder{idx_label}.pt"),
                               _use_new_zipfile_serialization=False)
                if self.base2final.task_decoder is not None:
                    torch.save(self.base2final.task_decoder.state_dict(), os.path.join(save_path, f"task_decoder{idx_label}.pt"),
                               _use_new_zipfile_serialization=False)
                if self.base2final.exploration_value_decoder is not None:
                    torch.save(self.base2final.exploration_value_decoder.state_dict(), os.path.join(save_path, f"exploration_value_decoder{idx_label}.pt"),
                               _use_new_zipfile_serialization=False)
                if self.base2final.exploitation_value_decoder is not None:
                    torch.save(self.base2final.exploitation_value_decoder.state_dict(), os.path.join(save_path, f"exploitation_value_decoder{idx_label}.pt"),
                               _use_new_zipfile_serialization=False)
                if self.base2final.action_decoder is not None:
                    torch.save(self.base2final.action_decoder.state_dict(), os.path.join(save_path, f"action_decoder{idx_label}.pt"),
                               _use_new_zipfile_serialization=False)
                tmp_dict = {
                    'iter_idx': self.iter_idx,
                    'vae_optimiser': self.base2final.optimiser_vae.state_dict(),
                }

                if self.exploration_policy is not None:
                    tmp_dict['exploration_policy_optimiser'] = self.exploration_policy.optimiser.state_dict()
                if self.exploitation_policy is not None:
                    tmp_dict['exploitation_policy_optimiser'] = self.exploitation_policy.optimiser.state_dict()

                torch.save(tmp_dict, os.path.join(save_path, f"general{idx_label}.pt"),
                           _use_new_zipfile_serialization=False)

                # save normalisation params of envs
                if self.args.norm_rew_for_policy:
                    rew_rms = envs.venv.ret_rms
                    filename = os.path.join(save_path, f"env_rew_rms_{policy_type}{idx_label}.pkl")
                    torch.save(rew_rms, filename, _use_new_zipfile_serialization=False)

                if self.state_prediction_running_normalizer is not None:
                    filename = os.path.join(save_path, f"state_error_rms{idx_label}.pkl")
                    torch.save(self.state_prediction_running_normalizer, filename, _use_new_zipfile_serialization=False)

                if self.action_prediction_running_normalizer is not None:
                    filename = os.path.join(save_path, f"action_error_rms{idx_label}.pkl")
                    torch.save(self.action_prediction_running_normalizer, filename, _use_new_zipfile_serialization=False)

                if self.args.norm_state_for_policy and self.args.pass_state_to_policy:
                    filename = os.path.join(save_path, f"policy_state_rms_{policy_type}{idx_label}.pkl")
                    torch.save(policy.actor_critic.state_rms, filename, _use_new_zipfile_serialization=False)

                if self.args.norm_task_inference_latent_for_policy and self.args.pass_task_inference_latent_to_policy:
                    filename = os.path.join(save_path, f"policy_latent_rms_{policy_type}{idx_label}.pkl")
                    torch.save(policy.actor_critic.task_inference_latent_rms, filename, _use_new_zipfile_serialization=False)

                if self.args.norm_rim_level1_output and self.args.use_rim_level1:
                    filename = os.path.join(save_path, f"policy_rim_level1_rms_{policy_type}{idx_label}.pkl")
                    torch.save(policy.actor_critic.rim_level1_output_rms, filename, _use_new_zipfile_serialization=False)

        # --- log some other things ---

        if (self.iter_idx % self.args.log_interval == 0) and (train_stats is not None):

            self.logger.add(f'{policy_type}_policy_losses/value_loss', train_stats[0], self.iter_idx)
            self.logger.add(f'{policy_type}_policy_losses/action_loss', train_stats[1], self.iter_idx)
            self.logger.add(f'{policy_type}_policy_losses/dist_entropy', train_stats[2], self.iter_idx)
            self.logger.add(f'{policy_type}_policy_losses/sum', train_stats[3], self.iter_idx)

            self.logger.add(f'{policy_type}_policy/action', run_stats[0][0].float().mean(), self.iter_idx)
            if hasattr(policy.actor_critic, 'logstd'):
                self.logger.add(f'{policy_type}_policy/action_logstd', policy.actor_critic.dist.logstd.mean(), self.iter_idx)
            self.logger.add(f'{policy_type}_policy/action_logprob', run_stats[1].mean(), self.iter_idx)
            self.logger.add(f'{policy_type}_policy/value', run_stats[2].mean(), self.iter_idx)

            self.logger.add('vae_encoder/latent_mean', torch.cat(policy_storage.latent_mean).mean(), self.iter_idx)
            self.logger.add('vae_encoder/latent_logvar', torch.cat(policy_storage.latent_logvar).mean(), self.iter_idx)

            # log the average weights and gradients of all models (where applicable)
            model_params = [
                [policy.actor_critic, 'policy'],
                [self.base2final.brim_core.brim.vae_encoder, 'vae_encoder'],
                [self.base2final.brim_core.brim.model, 'brim'],
                [self.base2final.reward_decoder, 'reward_decoder'],
                [self.base2final.state_decoder, 'state_transition_decoder'],
                [self.base2final.task_decoder, 'task_decoder'],
                [self.base2final.action_decoder, 'action_decoder'],
                [self.base2final.exploitation_value_decoder, 'exploitation_value_decoder'],
                [self.base2final.exploration_value_decoder, 'exploration_value_decoder'],
                [policy.actor_critic.state_encoder, f'state_encoder_{policy_type}'],
            ]
            if self.args.use_memory:
                model_params.append([self.base2final.brim_core.brim.model.memory, 'memory'])
            for [model, name] in model_params:
                if model is not None:
                    param_list = list(model.named_parameters())
                    param_mean = np.mean([param_list[i][1].data.cpu().numpy().mean() for i in range(len(param_list))])
                    self.logger.add('weights/{}'.format(name), param_mean, self.iter_idx)
                    if name == 'policy':
                        self.logger.add('weights/policy_std', param_list[0][1].data.mean(), self.iter_idx)
                    if param_list[0][1].grad is not None:
                        param_grad_mean = []
                        for i in range(len(param_list)):
                            if param_list[i][1].grad is not None:
                                param_grad_mean.append(param_list[i][1].grad.cpu().numpy().mean())
                            else:
                                pass
                        param_grad_mean = np.mean(param_grad_mean)
                        self.logger.add('gradients/{}'.format(name), param_grad_mean, self.iter_idx)



