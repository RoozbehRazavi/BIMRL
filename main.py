"""
Main scripts to start experiments.
Takes a flag --env-type (see below for choices) and loads the parameters from the respective config file.
"""
import argparse
import warnings
import gym_minigrid
import torch

# get configs
from config.gridworld import \
    args_grid_rl2, args_grid_varibad, args_base2final_exploration_rl_block,\
    args_base2final_exploration_full_model, args_base2final_exploration_BIMRL, args_base2final_exploration_memory_exploration, args_base2final_exploration_rl_attention, args_base2final_exploration_varibad
from config.mujoco import \
    args_cheetah_dir_oracle, args_cheetah_dir_rl2, args_cheetah_dir_varibad, \
    args_cheetah_vel_oracle, args_cheetah_vel_rl2, args_cheetah_vel_varibad, args_cheetah_vel_avg, \
    args_ant_dir_oracle, args_ant_dir_rl2, args_ant_dir_varibad, \
    args_ant_goal_oracle, args_ant_goal_rl2, args_ant_goal_varibad, \
    args_walker_oracle, args_walker_avg, args_walker_rl2, args_walker_varibad
from metalearner import MetaLearner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type', default='args_base2final_exploration_BIMRL')
    args, rest_args = parser.parse_known_args()
    env = args.env_type

    # --- GridWorld ---
    if env == 'gridworld_varibad':
        args = args_grid_varibad.get_args(rest_args)
    elif env == 'gridworld_rl2':
        args = args_grid_rl2.get_args(rest_args)
    elif env == 'gridworld_b2f_exploration_full_model':
        args = args_base2final_exploration_full_model.get_args(rest_args)
    elif env == 'args_base2final_exploration_BIMRL':
        args = args_base2final_exploration_BIMRL.get_args(rest_args)
    elif env == 'gridworld_b2f_exploration_memory_exploration':
        args = args_base2final_exploration_memory_exploration.get_args(rest_args)
    elif env == 'gridworld_b2f_exploration_rl_attention':
        args = args_base2final_exploration_rl_attention.get_args(rest_args)
    elif env == 'gridworld_b2f_exploration_varibad':
        args = args_base2final_exploration_varibad.get_args(rest_args)

    # --- MUJOCO ---

    # - AntDir -
    elif env == 'ant_dir_oracle':
        args = args_ant_dir_oracle.get_args(rest_args)
    elif env == 'ant_dir_rl2':
        args = args_ant_dir_rl2.get_args(rest_args)
    elif env == 'ant_dir_varibad':
        args = args_ant_dir_varibad.get_args(rest_args)
    #
    # - AntGoal -
    elif env == 'ant_goal_oracle':
        args = args_ant_goal_oracle.get_args(rest_args)
    elif env == 'ant_goal_varibad':
        args = args_ant_goal_varibad.get_args(rest_args)
    elif env == 'ant_goal_rl2':
        args = args_ant_goal_rl2.get_args(rest_args)
    #
    # - CheetahDir -
    elif env == 'cheetah_dir_oracle':
        args = args_cheetah_dir_oracle.get_args(rest_args)
    elif env == 'cheetah_dir_rl2':
        args = args_cheetah_dir_rl2.get_args(rest_args)
    elif env == 'cheetah_dir_varibad':
        args = args_cheetah_dir_varibad.get_args(rest_args)
    #
    # - CheetahVel -
    elif env == 'cheetah_vel_oracle':
        args = args_cheetah_vel_oracle.get_args(rest_args)
    elif env == 'cheetah_vel_rl2':
        args = args_cheetah_vel_rl2.get_args(rest_args)
    elif env == 'cheetah_vel_varibad':
        args = args_cheetah_vel_varibad.get_args(rest_args)
    elif env == 'cheetah_vel_avg':
        args = args_cheetah_vel_avg.get_args(rest_args)
    #
    # - Walker -
    elif env == 'walker_oracle':
        args = args_walker_oracle.get_args(rest_args)
    elif env == 'walker_avg':
        args = args_walker_avg.get_args(rest_args)
    elif env == 'walker_rl2':
        args = args_walker_rl2.get_args(rest_args)
    elif env == 'walker_varibad':
        args = args_walker_varibad.get_args(rest_args)

    # warning for deterministic execution
    if args.deterministic_execution:
        print('Envoking deterministic code execution.')
        if torch.backends.cudnn.enabled:
            warnings.warn('Running with deterministic CUDNN.')
        if args.num_processes > 1:
            raise RuntimeError('If you want fully deterministic code, use num_processes 1.'
                               'Warning: This will slow things down and might break A2C if '
                               'policy_num_steps < env._max_episode_steps.')

    # clean up arguments
    if hasattr(args, 'disable_decoder') and args.disable_decoder:
        args.decode_reward = False
        args.decode_state = False
        args.decode_task = False

    if hasattr(args, 'decode_only_past') and args.decode_only_past:
        args.split_batches_by_elbo = True
    # if hasattr(args, 'vae_subsample_decodes') and args.vae_subsample_decodes:
    #     args.split_batches_by_elbo = True

    # begin training (loop through all passed seeds)
    seed_list = [args.seed] if isinstance(args.seed, int) else args.seed
    for seed in seed_list:
        print('training', seed)
        args.seed = seed
        learner = MetaLearner(args)
        learner.train()


if __name__ == '__main__':
    main()
