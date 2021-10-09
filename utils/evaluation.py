import numpy as np
import torch

from environments.parallel_envs import make_vec_envs
from utils import helpers as utl

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def evaluate(args,
             policy,
             ret_rms,
             iter_idx,
             brim_core=None,
             num_episodes=None,
             policy_type='exploration'
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

    # --- initialise environments and latents ---

    envs = make_vec_envs(env_name, seed=args.seed * 42 + iter_idx, num_processes=num_processes,
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
        (latent_sample, latent_mean, latent_logvar, task_inference_hidden_state), = brim_core.prior(num_processes, sample=True)

        if policy_type == 'exploration':
            brim_output_level1 = brim_output1
            brim_output_level2 = brim_output3
        elif policy_type == 'exploitation':
            brim_output_level1 = brim_output2
            brim_output_level2 = brim_output4
        else:
            raise NotImplementedError
        brim_output_level3 = brim_output5
    else:
        latent_sample = latent_mean = latent_logvar = task_inference_hidden_state = None
        brim_output_level1 = brim_hidden_state = None

    for episode_idx in range(num_episodes):

        for step_idx in range(num_steps):

            with torch.no_grad():
                _, action, _ = utl.select_action(args=args,
                                                 policy=policy,
                                                 state=state,
                                                 belief=belief,
                                                 task=task,
                                                 latent_sample=latent_sample,
                                                 latent_mean=latent_mean,
                                                 latent_logvar=latent_logvar,
                                                 brim_output_level1=brim_output_level1,
                                                 deterministic=True)

            # observe reward and next obs
            [state, belief, task], (rew_raw, rew_normalised), done, infos = utl.env_step(envs, action, args)

            # replace intrinsic reward instead extrinsic reward
            if policy_type == 'exploration':
                rew_raw, rew_normalised = utl.compute_intrinsic_reward(rew_raw, rew_normalised)

            done_mdp = [info['done_mdp'] for info in infos]

            if brim_core is not None:
                # update the hidden state
                brim_output_level1, brim_output_level2, brim_output_level3, brim_hidden_state,\
                latent_sample, latent_mean, latent_logvar, task_inference_hidden_state = utl.update_encoding(brim_core=brim_core,
                                                                                                             next_obs=state,
                                                                                                             action=action,
                                                                                                             reward=rew_raw,
                                                                                                             done=None,
                                                                                                             task_inference_hidden_state=task_inference_hidden_state,
                                                                                                             brim_hidden_state=brim_hidden_state,
                                                                                                             activated_branch=policy_type)

            # add rewards
            returns_per_episode[range(num_processes), task_count] += rew_raw.view(-1)

            for i in np.argwhere(done_mdp).flatten():
                # count task up, but cap at num_episodes + 1
                task_count[i] = min(task_count[i] + 1, num_episodes)  # zero-indexed, so no +1
            if np.sum(done) > 0:
                done_indices = np.argwhere(done.flatten()).flatten()
                state, belief, task = utl.reset_env(envs, args, indices=done_indices, state=state)

    envs.close()

    return returns_per_episode[:, :num_episodes]