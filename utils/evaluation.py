import numpy as np
import torch
import matplotlib.pyplot as plt
from environments.parallel_envs import make_vec_envs
from utils import helpers as utl
from array2gif import write_gif
import math
import os

import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             state_decoder,
             action_decoder,
             reward_decoder,
             state_prediction_running_normalizer,
             action_prediction_running_normalizer,
             reward_prediction_running_normalizer,
             epi_reward_running_normalizer,
             brim_core,
             policy_type,
             num_updates,
             num_episodes=None,
             tmp=False,
             ):
    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name
    if num_episodes is None:
        num_episodes = args.max_rollouts_per_task

    if policy_type == 'exploration':
        num_processes = int(args.num_processes * args.exploration_processes_portion)

    if policy_type == 'exploitation':
        num_processes = int(args.num_processes * (1 - args.exploration_processes_portion))

    # --- set up the things we want to log ---

    # for each process, we log the returns during the first, second, ... episode
    # (such that we have a minimum of [num_episodes]; the last column is for
    #  any overflow and will be discarded at the end, because we need to wait until
    #  all processes have at least [num_episodes] many episodes)
    returns_per_episode = torch.zeros((num_processes, num_episodes + 1)).to(device)
    returns_per_episode__ = torch.zeros((num_processes, num_episodes + 1)).to(device)

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name, seed=int(args.seed * 1e6 + iter_idx), num_processes=num_processes,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=num_processes + 1,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy, ret_rms=ret_rms)
    num_steps = envs._max_episode_steps

    # reset environments
    state, belief, task = utl.reset_env(envs, args)

    # this counts how often an agent has done the same task already
    task_count = torch.zeros(num_processes).long().to(device)

    if brim_core is not None:
        # reset latent state to prior
        (brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state),\
        (latent_sample, latent_mean, latent_logvar, task_inference_hidden_state), policy_embedded_state = brim_core.prior(
            policy_network=policy.actor_critic,
            batch_size=num_processes,
            state=state,
            sample=True,
            activated_branch=policy_type)

        if policy_type == 'exploration':
            brim_output_level1 = brim_output1
            brim_output_level2 = brim_output3
            if 'exploration_policy_embedded_state' in policy_embedded_state:
                policy_embedded_state = policy_embedded_state['exploration_policy_embedded_state']
        elif policy_type == 'exploitation':
            brim_output_level1 = brim_output2
            brim_output_level2 = brim_output4
            if 'exploitation_policy_embedded_state' in policy_embedded_state:
                policy_embedded_state = policy_embedded_state['exploitation_policy_embedded_state']
        else:
            raise NotImplementedError
        brim_output_level3 = brim_output5
    else:
        latent_sample = latent_mean = latent_logvar = task_inference_hidden_state = None
        brim_output_level1 = brim_hidden_state = None
        policy_embedded_state = None

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):

            with torch.no_grad():
                _, action, _ = utl.select_action(args=args,
                                                 policy=policy,
                                                 belief=belief,
                                                 task=task,
                                                 latent_sample=latent_sample,
                                                 latent_mean=latent_mean,
                                                 latent_logvar=latent_logvar,
                                                 brim_output_level1=brim_output_level1,
                                                 policy_embedded_state=policy_embedded_state,
                                                 deterministic=True)
            prev_state = state

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            done_ = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
            rew_raw__ = rew_raw

            # replace intrinsic reward instead extrinsic reward
            if policy_type == 'exploration':
                latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                   add_nonlinearity_to_latent=args.add_nonlinearity_to_latent,
                                                   latent_sample=latent_sample,
                                                   latent_mean=latent_mean,
                                                   latent_logvar=latent_logvar)
                memory_latent = utl.get_latent_for_policy(sample_embeddings=False,
                                                          add_nonlinearity_to_latent=False,
                                                          latent_sample=latent_sample,
                                                          latent_mean=latent_mean,
                                                          latent_logvar=latent_logvar)

                if args.use_rim_level3:
                    if args.residual_task_inference_latent:
                        latent = torch.cat((brim_output_level3.squeeze(0), latent), dim=-1)
                    else:
                        latent = brim_output_level3

                rew_raw, rew_normalised, _, _, _ = utl.compute_intrinsic_reward(rew_raw=rew_raw,
                                                                             rew_normalised=rew_normalised,
                                                                             latent=latent,
                                                                             prev_state=prev_state,
                                                                             next_state=state,
                                                                             action=action.float(),
                                                                             state_decoder=state_decoder,
                                                                             action_decoder=action_decoder,
                                                                             decode_action=args.decode_action,
                                                                             state_prediction_running_normalizer=state_prediction_running_normalizer,
                                                                             action_prediction_running_normalizer=action_prediction_running_normalizer,
                                                                             state_prediction_intrinsic_reward_coef=args.state_prediction_intrinsic_reward_coef,
                                                                             action_prediction_intrinsic_reward_coef=args.action_prediction_intrinsic_reward_coef,
                                                                             extrinsic_reward_intrinsic_reward_coef=args.extrinsic_reward_intrinsic_reward_coef,
                                                                             reward_decoder=reward_decoder,
                                                                             rew_pred_type=args.rew_pred_type,
                                                                             reward_prediction_running_normalizer=reward_prediction_running_normalizer,
                                                                             reward_prediction_intrinsic_reward_coef=args.reward_prediction_intrinsic_reward_coef,
                                                                             decode_reward=args.decode_reward,
                                                                             itr_idx=iter_idx,
                                                                             num_updates=num_updates,
                                                                             memory=brim_core.brim.model.memory,
                                                                             episodic_reward=args.episodic_reward,
                                                                             episodic_reward_coef=args.episodic_reward_coef,
                                                                             task_inf_latent=memory_latent,
                                                                             epi_reward_running_normalizer=epi_reward_running_normalizer
                                                                            )

            done_mdp = list()
            for i in range(num_processes):
                done_mdp.append(1.0 if infos[i]['done_mdp'] else 0.0)
            done_mdp = torch.Tensor(done_mdp).float().to(device).unsqueeze(1)

            if brim_core is not None:
                # update the hidden state
                brim_output_level1, brim_output_level2, brim_output_level3, brim_hidden_state,\
                latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, policy_embedded_state = utl.update_encoding(
                    brim_core=brim_core,
                    policy=policy.actor_critic,
                    next_obs=state,
                    action=action,
                    reward=rew_raw,
                    done=done_,
                    task_inference_hidden_state=task_inference_hidden_state,
                    brim_hidden_state=brim_hidden_state,
                    activated_branch=policy_type,
                    done_episode=done_mdp,
                    rpe=None)

            # add rewards
            if tmp:
                returns_per_episode__[range(num_processes), task_count] += rew_raw__.view(-1)
                returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            done_mdp = [info['done_mdp'] for info in infos]
            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

    envs.close()

    return returns_per_episode[:, :num_episodes], returns_per_episode__[:, :num_episodes]


def plot_meta_eval(returns, save_path, iter_idx):
    plt.plot(range(int(returns.shape[1])), returns[0], linestyle='-', marker='o', alpha=0.5)
    plt.xticks(range(int(returns.shape[1])))
    plt.xlabel('exploration episode', fontsize=15)
    plt.ylabel('exploitation mean return', fontsize=15)
    plt.tight_layout()
    if save_path is not None:
        if os.path.isfile(os.path.join(save_path, f'meta_eval_{iter_idx}.png')):
            os.remove(os.path.join(save_path, f'meta_eval_{iter_idx}.png'))
        plt.savefig(os.path.join(save_path, f'meta_eval_{iter_idx}'))
        plt.close()
    else:
        plt.show()


def evaluate_meta_policy(
        args,
        exploration_policy,
        exploitation_policy,
        ret_rms,
        iter_idx,
        state_decoder,
        action_decoder,
        reward_decoder,
        state_prediction_running_normalizer,
        action_prediction_running_normalizer,
        reward_prediction_running_normalizer,
        epi_reward_running_normalizer,
        brim_core,
        exploration_num_episodes,
        save_path,
        num_updates):

    env_name = args.env_name
    if hasattr(args, 'test_env_name'):
        env_name = args.test_env_name

    # only for last episode
    num_processes = args.num_processes
    returns_per_episode = torch.zeros((num_processes, exploration_num_episodes)).to(device)
    # --- initialise environments and latents ---
    for i in range(exploration_num_episodes):
        envs = make_vec_envs(env_name, seed=int(args.seed * 1e6 + iter_idx), num_processes=num_processes,
                             gamma=args.policy_gamma,
                             device=device,
                             rank_offset=2,  # to use diff tmp folders than main processes
                             episodes_per_task=i + 1,
                             normalise_rew=args.norm_rew_for_policy, ret_rms=ret_rms)
        num_steps = envs._max_episode_steps

        # reset environments
        state, belief, task = utl.reset_env(envs, args)
        if state.shape[-1] == 147:
            state = torch.cat((state, torch.zeros((num_processes, 1), device=device)), dim=-1)
        # this counts how often an agent has done the same task already

        activated_branch = 'exploration'
        policy = exploration_policy
        if brim_core is not None:
            # reset latent state to prior
            (brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state),\
            (latent_sample, latent_mean, latent_logvar, task_inference_hidden_state), policy_embedded_state = brim_core.prior(
                policy_network=policy.actor_critic,
                batch_size=num_processes,
                state=state,
                sample=True,
                activated_branch=activated_branch)

            if activated_branch == 'exploration':
                brim_output_level1 = brim_output1
                brim_output_level2 = brim_output3
                if 'exploration_policy_embedded_state' in policy_embedded_state:
                    policy_embedded_state = policy_embedded_state['exploration_policy_embedded_state']
            elif activated_branch == 'exploitation':
                brim_output_level1 = brim_output2
                brim_output_level2 = brim_output4
                if 'exploitation_policy_embedded_state' in policy_embedded_state:
                    policy_embedded_state = policy_embedded_state['exploitation_policy_embedded_state']
            else:
                raise NotImplementedError
            brim_output_level3 = brim_output5
        else:
            latent_sample = latent_mean = latent_logvar = task_inference_hidden_state = None
            brim_output_level1 = brim_hidden_state = None
            policy_embedded_state = None

        for episode_idx in range(i + 1):

            if episode_idx == i:
                activated_branch = 'exploitation'
                policy = exploitation_policy
                if brim_core is not None:
                    # reset latent state to prior
                    (brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state), \
                    (latent_sample, latent_mean, latent_logvar,
                     task_inference_hidden_state), policy_embedded_state = brim_core.prior(
                        policy_network=policy.actor_critic,
                        batch_size=num_processes,
                        state=state,
                        sample=True,
                        activated_branch=activated_branch)

                    if activated_branch == 'exploration':
                        brim_output_level1 = brim_output1
                        brim_output_level2 = brim_output3
                        if 'exploration_policy_embedded_state' in policy_embedded_state:
                            policy_embedded_state = policy_embedded_state['exploration_policy_embedded_state']
                    elif activated_branch == 'exploitation':
                        brim_output_level1 = brim_output2
                        brim_output_level2 = brim_output4
                        if 'exploitation_policy_embedded_state' in policy_embedded_state:
                            policy_embedded_state = policy_embedded_state['exploitation_policy_embedded_state']
                    else:
                        raise NotImplementedError
                    brim_output_level3 = brim_output5
                else:
                    latent_sample = latent_mean = latent_logvar = task_inference_hidden_state = None
                    brim_output_level1 = brim_hidden_state = None
                    policy_embedded_state = None

            for step_idx in range(num_steps):
                with torch.no_grad():
                    _, action, _ = utl.select_action(args=args,
                                                     policy=policy,
                                                     belief=belief,
                                                     task=task,
                                                     latent_sample=latent_sample,
                                                     latent_mean=latent_mean,
                                                     latent_logvar=latent_logvar,
                                                     brim_output_level1=brim_output_level1,
                                                     policy_embedded_state=policy_embedded_state,
                                                     deterministic=True)
                prev_state = state

                # observe reward and next obs
                [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
                done_ = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))
                if state.shape[-1] == 147:
                    state = torch.cat((state, torch.zeros((num_processes, 1), device=device)), dim=-1)

                # replace intrinsic reward instead extrinsic reward
                if activated_branch == 'exploration':
                    latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                       add_nonlinearity_to_latent=args.add_nonlinearity_to_latent,
                                                       latent_sample=latent_sample,
                                                       latent_mean=latent_mean,
                                                       latent_logvar=latent_logvar)
                    memory_latent = utl.get_latent_for_policy(sample_embeddings=False,
                                                              add_nonlinearity_to_latent=False,
                                                              latent_sample=latent_sample,
                                                              latent_mean=latent_mean,
                                                              latent_logvar=latent_logvar)

                    if args.use_rim_level3:
                        if args.residual_task_inference_latent:
                            if brim_output_level3.dim() == 3:
                                brim_output_level3 = brim_output_level3.squeeze(0)
                            latent = torch.cat((brim_output_level3, latent), dim=-1)
                        else:
                            latent = brim_output_level3

                    rew_raw, rew_normalised, _, _, _ = utl.compute_intrinsic_reward(rew_raw=rew_raw,
                                                                                 rew_normalised=rew_normalised,
                                                                                 latent=latent,
                                                                                 prev_state=prev_state,
                                                                                 next_state=state,
                                                                                 action=action.float(),
                                                                                 state_decoder=state_decoder,
                                                                                 action_decoder=action_decoder,
                                                                                 decode_action=args.decode_action,
                                                                                 state_prediction_running_normalizer=state_prediction_running_normalizer,
                                                                                 action_prediction_running_normalizer=action_prediction_running_normalizer,
                                                                                 state_prediction_intrinsic_reward_coef=args.state_prediction_intrinsic_reward_coef,
                                                                                 action_prediction_intrinsic_reward_coef=args.action_prediction_intrinsic_reward_coef,
                                                                                 extrinsic_reward_intrinsic_reward_coef=args.extrinsic_reward_intrinsic_reward_coef,
                                                                                 reward_decoder=reward_decoder,
                                                                                 rew_pred_type=args.rew_pred_type,
                                                                                 reward_prediction_running_normalizer=reward_prediction_running_normalizer,
                                                                                 reward_prediction_intrinsic_reward_coef=args.reward_prediction_intrinsic_reward_coef,
                                                                                 decode_reward=args.decode_reward,
                                                                                 itr_idx=iter_idx,
                                                                                 num_updates=num_updates,
                                                                                 memory=brim_core.brim.model.memory,
                                                                                 episodic_reward=args.episodic_reward,
                                                                                 episodic_reward_coef=args.episodic_reward_coef,
                                                                                 task_inf_latent=memory_latent,
                                                                                 epi_reward_running_normalizer=epi_reward_running_normalizer
                                                                                 )

                done_mdp = list()
                for i in range(num_processes):
                    done_mdp.append(1.0 if infos[i]['done_mdp'] else 0.0)
                done_mdp = torch.Tensor(done_mdp).float().to(device).unsqueeze(1)

                if brim_core is not None:
                    # update the hidden state
                    brim_output_level1, brim_output_level2, brim_output_level3, brim_hidden_state,\
                    latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, policy_embedded_state = utl.update_encoding(
                        brim_core=brim_core,
                        policy=policy.actor_critic,
                        next_obs=state,
                        action=action,
                        reward=rew_raw,
                        done=done_,
                        task_inference_hidden_state=task_inference_hidden_state,
                        brim_hidden_state=brim_hidden_state,
                        activated_branch=activated_branch,
                        done_episode=done_mdp,
                        rpe=None)

                # add rewards
                if episode_idx == i:
                    returns_per_episode[:, i:i+1] += rew_raw
                if sum(done_mdp) > 0:
                    break
        envs.close()

    plot_meta_eval(returns_per_episode.mean(0).unsqueeze(0).detach().cpu(), save_path, iter_idx)


def visualize_policy(
        args,
        policy,
        ret_rms,
        brim_core,
        iter_idx,
        policy_type,
        state_decoder,
        action_decoder,
        reward_decoder,
        num_episodes,
        state_prediction_running_normalizer,
        action_prediction_running_normalizer,
        reward_prediction_running_normalizer,
        epi_reward_running_normalizer,
        full_output_folder,
        num_updates):

    env_name = args.env_name
    envs = make_vec_envs(env_name, seed=int(args.seed * 1e6 + iter_idx), num_processes=1,
                         gamma=args.policy_gamma,
                         device=device,
                         rank_offset=2,  # to use diff tmp folders than main processes
                         episodes_per_task=num_episodes,
                         normalise_rew=args.norm_rew_for_policy, ret_rms=ret_rms)
    num_steps = envs._max_episode_steps

    state, belief, task = utl.reset_env(envs, args)
    if state.shape[-1] == 147:
        state = torch.cat((state, torch.zeros((1, 1), device=device)), dim=-1)
    frames = []

    if brim_core is not None:
        # reset latent state to prior
        (brim_output1, brim_output2, brim_output3, brim_output4, brim_output5, brim_hidden_state), \
        (latent_sample, latent_mean, latent_logvar,
         task_inference_hidden_state), policy_embedded_state = brim_core.prior(
            policy_network=policy.actor_critic,
            batch_size=1,
            state=state,
            sample=True,
            activated_branch=policy_type)

        if policy_type == 'exploration':
            brim_output_level1 = brim_output1
            brim_output_level2 = brim_output3
            if 'exploration_policy_embedded_state' in policy_embedded_state:
                policy_embedded_state = policy_embedded_state['exploration_policy_embedded_state']
        elif policy_type == 'exploitation':
            brim_output_level1 = brim_output2
            brim_output_level2 = brim_output4
            if 'exploitation_policy_embedded_state' in policy_embedded_state:
                policy_embedded_state = policy_embedded_state['exploitation_policy_embedded_state']
        else:
            raise NotImplementedError
        brim_output_level3 = brim_output5
    else:
        latent_sample = latent_mean = latent_logvar = task_inference_hidden_state = None
        brim_output_level1 = brim_hidden_state = None
        policy_embedded_state = None

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):
            a = envs.render("rgb_array", True)
            frames.append(numpy.moveaxis(a, 2, 0))
            with torch.no_grad():
                _, action, _ = utl.select_action(args=args,
                                                 policy=policy,
                                                 belief=belief,
                                                 task=task,
                                                 latent_sample=latent_sample,
                                                 latent_mean=latent_mean,
                                                 latent_logvar=latent_logvar,
                                                 brim_output_level1=brim_output_level1,
                                                 policy_embedded_state=policy_embedded_state,
                                                 deterministic=True)
            prev_state = state

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)
            if state.shape[-1] == 147:
                state = torch.cat((state, torch.zeros((1, 1), device=device)), dim=-1)
            done_ = torch.from_numpy(np.array(done, dtype=int)).to(device).float().view((-1, 1))

            # replace intrinsic reward instead extrinsic reward
            if policy_type == 'exploration':
                latent = utl.get_latent_for_policy(sample_embeddings=True,
                                                   add_nonlinearity_to_latent=args.add_nonlinearity_to_latent,
                                                   latent_sample=latent_sample,
                                                   latent_mean=latent_mean,
                                                   latent_logvar=latent_logvar)
                memory_latent = utl.get_latent_for_policy(sample_embeddings=False,
                                                          add_nonlinearity_to_latent=False,
                                                          latent_sample=latent_sample,
                                                          latent_mean=latent_mean,
                                                          latent_logvar=latent_logvar)
                if args.use_rim_level3:
                    if args.residual_task_inference_latent:
                        if brim_output_level3.dim() == 3:
                            brim_output_level3 = brim_output_level3.squeeze(1)
                        latent = torch.cat((brim_output_level3, latent), dim=-1)
                    else:
                        latent = brim_output_level3

                rew_raw, rew_normalised, _, _, _ = utl.compute_intrinsic_reward(
                    rew_raw=rew_raw,
                    rew_normalised=rew_normalised,
                    latent=latent,
                    prev_state=prev_state,
                    next_state=state,
                    action=action.float(),
                    state_decoder=state_decoder,
                    action_decoder=action_decoder,
                    decode_action=args.decode_action,
                    state_prediction_running_normalizer=state_prediction_running_normalizer,
                    action_prediction_running_normalizer=action_prediction_running_normalizer,
                    state_prediction_intrinsic_reward_coef=args.state_prediction_intrinsic_reward_coef,
                    action_prediction_intrinsic_reward_coef=args.action_prediction_intrinsic_reward_coef,
                    extrinsic_reward_intrinsic_reward_coef=args.extrinsic_reward_intrinsic_reward_coef,
                    reward_decoder=reward_decoder,
                    rew_pred_type=args.rew_pred_type,
                    reward_prediction_running_normalizer=reward_prediction_running_normalizer,
                    reward_prediction_intrinsic_reward_coef=args.reward_prediction_intrinsic_reward_coef,
                    decode_reward=args.decode_reward,
                    itr_idx=iter_idx,
                    num_updates=num_updates,
                    memory=brim_core.brim.model.memory,
                    episodic_reward=args.episodic_reward,
                    episodic_reward_coef=args.episodic_reward_coef,
                    task_inf_latent=memory_latent,
                    epi_reward_running_normalizer=epi_reward_running_normalizer
                )

            done_mdp = list()
            for i in range(1):
                done_mdp.append(1.0 if infos[i]['done_mdp'] else 0.0)
            done_mdp = torch.Tensor(done_mdp).float().to(device).unsqueeze(1)

            if brim_core is not None:
                # update the hidden state
                brim_output_level1, brim_output_level2, brim_output_level3, brim_hidden_state,\
                latent_sample, latent_mean, latent_logvar, task_inference_hidden_state, policy_embedded_state = utl.update_encoding(
                    brim_core=brim_core,
                    policy=policy.actor_critic,
                    next_obs=state,
                    action=action,
                    reward=rew_raw,
                    done=done_,
                    task_inference_hidden_state=task_inference_hidden_state,
                    brim_hidden_state=brim_hidden_state,
                    activated_branch=policy_type,
                    done_episode=done_mdp,
                    rpe=None)

            if sum(done_mdp) == 1:
                break
    envs.close()
    save_path = os.path.join(full_output_folder, f'{policy_type}_policy_{iter_idx}.gif')
    if os.path.isfile(os.path.join(save_path, f'{policy_type}_policy_{iter_idx}.gif')):
        os.remove(os.path.join(save_path, f'{policy_type}_policy_{iter_idx}.gif'))
    write_gif(numpy.array(frames), save_path)
