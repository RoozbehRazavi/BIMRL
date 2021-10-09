import argparse
from utils.helpers import boolean_argument


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # --- GENERAL ---

    parser.add_argument('--num_frames', type=int, default=1e8, help='number of frames to train')
    parser.add_argument('--max_rollouts_per_task', type=int, default=4, help='number of MDP episodes for adaptation')
    parser.add_argument('--exp_label', default='varibad', help='label (typically name of method)')
    parser.add_argument('--env_name', default='MiniGrid-Empty-5x5-v0', help='environment to train on')

    # --- POLICY ---

    # what to pass to the policy (note this is after the encoder)
    parser.add_argument('--pass_state_to_policy', type=boolean_argument, default=True, help='condition policy on state')
    parser.add_argument('--pass_task_inference_latent_to_policy', type=boolean_argument, default=True, help='condition policy on VAE latent')
    parser.add_argument('--pass_belief_to_policy', type=boolean_argument, default=False, help='condition policy on ground-truth belief')
    parser.add_argument('--pass_task_to_policy', type=boolean_argument, default=False, help='condition policy on ground-truth task description')

    # using separate encoders for the different inputs ("None" uses no encoder)
    parser.add_argument('--policy_state_embedding_dim', type=int, default=16)
    parser.add_argument('--policy_task_inference_latent_embedding_dim', type=int, default=16)
    parser.add_argument('--policy_belief_embedding_dim', type=int, default=None)
    parser.add_argument('--policy_task_embedding_dim', type=int, default=None)

    # normalising (inputs/rewards/outputs)
    parser.add_argument('--norm_state_for_policy', type=boolean_argument, default=True, help='normalise state input')
    parser.add_argument('--norm_task_inference_latent_for_policy', type=boolean_argument, default=True, help='normalise latent input')
    parser.add_argument('--norm_belief_for_policy', type=boolean_argument, default=True, help='normalise belief input')
    parser.add_argument('--norm_task_for_policy', type=boolean_argument, default=True, help='normalise task input')
    parser.add_argument('--norm_rew_for_policy', type=boolean_argument, default=True, help='normalise rew for RL train')
    parser.add_argument('--norm_actions_of_policy', type=boolean_argument, default=True, help='normalise policy output')

    # network
    parser.add_argument('--policy_layers', nargs='+', default=[32])
    parser.add_argument('--policy_activation_function', type=str, default='tanh', help='tanh/relu/leaky-relu')
    parser.add_argument('--policy_initialisation', type=str, default='normc', help='normc/orthogonal')
    parser.add_argument('--policy_anneal_lr', type=boolean_argument, default=False)

    # RL algorithm
    parser.add_argument('--policy', type=str, default='ppo', help='choose: a2c, ppo')
    parser.add_argument('--policy_optimiser', type=str, default='adam', help='choose: rmsprop, adam')

    # PPO specific
    parser.add_argument('--ppo_num_epochs', type=int, default=2, help='number of epochs per PPO update')
    parser.add_argument('--ppo_num_minibatch', type=int, default=4, help='number of minibatches to split the data')
    parser.add_argument('--ppo_use_huberloss', type=boolean_argument, default=True, help='use huberloss instead of MSE')
    parser.add_argument('--ppo_use_clipped_value_loss', type=boolean_argument, default=True, help='clip value loss')
    parser.add_argument('--ppo_clip_param', type=float, default=0.05, help='clamp param')

    # other hyperparameters
    parser.add_argument('--lr_policy', type=float, default=0.0007, help='learning rate (default: 7e-4)')
    parser.add_argument('--num_processes', type=int, default=2,
                        help='how many training CPU processes / parallel environments to use (default: 16)')
    parser.add_argument('--policy_num_steps', type=int, default=400,
                        help='number of env steps to do (per process) before updating')
    parser.add_argument('--policy_eps', type=float, default=1e-5, help='optimizer epsilon (1e-8 for ppo, 1e-5 for a2c)')
    parser.add_argument('--policy_init_std', type=float, default=1.0, help='only used for continuous actions')
    parser.add_argument('--policy_value_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--policy_entropy_coef', type=float, default=0.1, help='entropy term coefficient')
    parser.add_argument('--policy_gamma', type=float, default=0.95, help='discount factor for rewards')
    parser.add_argument('--policy_use_gae', type=boolean_argument, default=True,
                        help='use generalized advantage estimation')
    parser.add_argument('--policy_tau', type=float, default=0.95, help='gae parameter')
    parser.add_argument('--use_proper_time_limits', type=boolean_argument, default=False,
                        help='treat timeout and death differently (important in mujoco)')
    parser.add_argument('--policy_max_grad_norm', type=float, default=0.5, help='max norm of gradients')

    # --- VAE TRAINING ---

    # general
    parser.add_argument('--lr_vae', type=float, default=0.001)
    parser.add_argument('--size_vae_buffer', type=int, default=1000,
                        help='how many trajectories (!) to keep in VAE buffer')
    parser.add_argument('--precollect_len', type=int, default=500,
                        help='how many frames to pre-collect before training begins (useful to fill VAE buffer)')
    parser.add_argument('--vae_buffer_add_thresh', type=float, default=1,
                        help='probability of adding a new trajectory to buffer')
    parser.add_argument('--vae_batch_num_trajs', type=int, default=25,
                        help='how many trajectories to use for VAE update')
    parser.add_argument('--tbptt_stepsize', type=int, default=None,
                        help='stepsize for truncated backpropagation through time; None uses max (horizon of BAMDP)')
    parser.add_argument('--vae_subsample_elbos', type=int, default=100,
                        help='for how many timesteps to compute the ELBO; None uses all')
    parser.add_argument('--vae_subsample_decodes', type=int, default=100,
                        help='number of reconstruction terms to subsample; None uses all')
    parser.add_argument('--vae_avg_elbo_terms', type=boolean_argument, default=False,
                        help='Average ELBO terms (instead of sum)')
    parser.add_argument('--vae_avg_reconstruction_terms', type=boolean_argument, default=False,
                        help='Average reconstruction terms (instead of sum)')
    parser.add_argument('--num_vae_updates', type=int, default=3,
                        help='how many VAE update steps to take per meta-iteration')
    parser.add_argument('--pretrain_len', type=int, default=0, help='for how many updates to pre-train the VAE')
    parser.add_argument('--kl_weight', type=float, default=1, help='weight for the KL term')

    parser.add_argument('--split_batches_by_task', type=boolean_argument, default=False,
                        help='split batches up by task (to save memory or if tasks are of different length)')
    parser.add_argument('--split_batches_by_elbo', type=boolean_argument, default=False,
                        help='split batches up by elbo term (to save memory of if ELBOs are of different length)')

    # - encoder
    parser.add_argument('--state_embedding_size', type=int, default=32)
    parser.add_argument('--vae_encoder_layers_before_gru', nargs='+', type=int, default=[])
    parser.add_argument('--vae_encoder_gru_hidden_size', type=int, default=64, help='dimensionality of RNN hidden state')
    parser.add_argument('--vae_encoder_layers_after_gru', nargs='+', type=int, default=[])
    parser.add_argument('--task_inference_latent_dim', type=int, default=5, help='dimensionality of latent space')

    # - decoder: rewards
    parser.add_argument('--decode_reward', type=boolean_argument, default=True, help='use reward decoder')
    parser.add_argument('--rew_loss_coeff', type=float, default=1.0, help='weight for state loss (vs reward loss)')
    parser.add_argument('--input_prev_state', type=boolean_argument, default=False, help='use prev state for rew pred')
    parser.add_argument('--reward_decoder_layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--multihead_for_reward', type=boolean_argument, default=False,
                        help='one head per reward pred (i.e. per state)')
    parser.add_argument('--rew_pred_type', type=str, default='bernoulli',
                        help='choose: '
                             'bernoulli (predict p(r=1|s))'
                             'categorical (predict p(r=1|s) but use softmax instead of sigmoid)'
                             'deterministic (treat as regression problem)')

    # - decoder: state transitions
    parser.add_argument('--decode_state', type=boolean_argument, default=True, help='use state decoder')
    parser.add_argument('--state_loss_coeff', type=float, default=1.0, help='weight for state loss')
    parser.add_argument('--state_decoder_layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--state_pred_type', type=str, default='deterministic', help='choose: deterministic, gaussian')

    # - decoder: ground-truth task ("varibad oracle", after Humplik et al. 2019)
    parser.add_argument('--decode_task', type=boolean_argument, default=False, help='use task decoder')
    parser.add_argument('--task_loss_coeff', type=float, default=1.0, help='weight for task loss')
    parser.add_argument('--task_decoder_layers', nargs='+', type=int, default=[32, 32])
    parser.add_argument('--task_pred_type', type=str, default='task_id', help='choose: task_id, task_description')

    # --- ABLATIONS ---
    parser.add_argument('--add_nonlinearity_to_latent', type=boolean_argument, default=False,
                        help='Use relu before feeding latent to policy')
    parser.add_argument('--disable_decoder', type=boolean_argument, default=False,
                        help='train without decoder')
    parser.add_argument('--disable_stochasticity_in_latent', type=boolean_argument, default=False,
                        help='use auto-encoder (non-variational)')
    parser.add_argument('--sample_embeddings', type=boolean_argument, default=False,
                        help='sample embedding for policy, instead of full belief')
    parser.add_argument('--vae_loss_coeff', type=float, default=1.0,
                        help='weight for VAE loss (vs RL loss)')
    parser.add_argument('--kl_to_gauss_prior', type=boolean_argument, default=False,
                        help='KL term in ELBO to fixed Gaussian prior (instead of prev approx posterior)')
    parser.add_argument('--decode_only_past', type=boolean_argument, default=False,
                        help='only decoder past observations, not the future')
    parser.add_argument('--condition_policy_on_state', type=boolean_argument, default=True,
                        help='after the encoder, concatenate env state and latent variable')

    # --- OTHERS ---

    # logging, saving, evaluation
    parser.add_argument('--log_interval', type=int, default=1, help='log interval, one log per n updates')
    parser.add_argument('--save_interval', type=int, default=1000, help='save interval, one save per n updates')
    parser.add_argument('--save_intermediate_models', type=boolean_argument, default=False, help='save all models')
    parser.add_argument('--eval_interval', type=int, default=1, help='eval interval, one eval per n updates')
    parser.add_argument('--vis_interval', type=int, default=500, help='visualisation interval, one eval per n updates')
    parser.add_argument('--results_log_dir', default=None, help='directory to save results (None uses ./logs)')

    # general settings
    parser.add_argument('--seed',  nargs='+', type=int, default=[73])
    parser.add_argument('--deterministic_execution', type=boolean_argument, default=False,
                        help='Make code fully deterministic. Expects 1 process and uses deterministic CUDNN')

    parser.add_argument('--load_model', type=boolean_argument, default=False, )

    # General Base2Final
    parser.add_argument('--vae_fill_with_exploration_experience', type=boolean_argument, default=False,
                        help='vae buffer fill just with exploration trajectory of with both exploration and exploitation')

    parser.add_argument('--exploration_processes_portion', type=float, default=0.0,
                        help='what portion of process generate trajectory with exploration policy')

    # Disable Loss of Base2Final

    parser.add_argument('--rlloss_through_encoder', type=boolean_argument, default=True,
                        help='backprop rl loss through encoder')
    parser.add_argument('--n_step_state_prediction', type=boolean_argument, default=True,
                        help='state prediction for n step forward not just next state')
    parser.add_argument('--n_step_reward_prediction', type=boolean_argument, default=True,
                        help='reward prediction for n step forward not just next reward')
    parser.add_argument('--n_prediction', type=int, default=2,
                        help='for how many future step state prediction should does (exclude one step prediction; for til t+3 set to 2)')

    # Coefficient in Base2Final
    parser.add_argument('--add_extrinsic_reward_to_intrinsic', type=boolean_argument, default=True,
                        help='for compute intrinsic reward add extrinsic also')

    parser.add_argument('--vae_avg_n_step_prediction', type=boolean_argument, default=False,
                        help='Average n step prediction terms (instead of sum)')

    parser.add_argument('--action_embedding_size', type=int, default=8)

    parser.add_argument('--reward_embedding_size', type=int, default=16)

    parser.add_argument('--action_simulator_hidden_size', type=int, default=16,
                        help='hidden size of GRU used in n step state prediction')
    parser.add_argument('--reward_simulator_hidden_size', type=int, default=16,
                        help='hidden size of GRU used in n step reward prediction')
    parser.add_argument('--input_action', type=boolean_argument, default=True, help='use prev action for rew pred')


    # RIM configuration
    parser.add_argument('--use_gru_or_rim', type=str, default='RIM',
                        help='as a RNN model use RIM or GRU')

    parser.add_argument('--use_rim_level1', type=boolean_argument, default=True,
                        help='whatever create rim level1 (use for policy) or not')

    parser.add_argument('--use_rim_level2', type=boolean_argument, default=False,
                        help='whatever create rim level2 (use for n step value prediction) or not')

    parser.add_argument('--use_rim_level3', type=boolean_argument, default=True,
                        help='whatever create rim level3 (use for decode VAE terms) or not')

    parser.add_argument('--rim_level1_hidden_size', type=int, default=32,
                        help='hidden size of level 1 rim (output of this level use for policy head)')
    parser.add_argument('--rim_level2_hidden_size', type=int, default=32,
                        help='hidden size of level 1 rim (output of this level use for n step value prediction head)')
    parser.add_argument('--rim_level3_hidden_size', type=int, default=32,
                        help='hidden size of level 3 rim (output of this level use decode VAE term)')

    parser.add_argument('--rim_level1_num_modules', type=int, default=1,
                        help='number of module in rim level 1')
    parser.add_argument('--rim_level2_num_modules', type=int, default=4,
                        help='number of module in rim level 2')
    parser.add_argument('--rim_level3_num_modules', type=int, default=4,
                        help='number of module in rim level 3')

    parser.add_argument('--rim_level1_topk', type=int, default=1,
                        help='number of module in rim level 1 that can active in each time step')
    parser.add_argument('--rim_level2_topk', type=int, default=3,
                        help='number of module in rim level 2 that can active in each time step')
    parser.add_argument('--rim_level3_topk', type=int, default=3,
                        help='number of module in rim level 3 that can active in each time step')

    parser.add_argument('--brim_layers_before_rim_level1', nargs='+', type=int, default=[16])
    parser.add_argument('--brim_layers_before_rim_level2', nargs='+', type=int, default=[16])
    parser.add_argument('--brim_layers_before_rim_level3', nargs='+', type=int, default=[16])

    parser.add_argument('--brim_layers_after_rim_level1', nargs='+', type=int, default=[8])
    parser.add_argument('--brim_layers_after_rim_level2', nargs='+', type=int, default=[8])
    parser.add_argument('--brim_layers_after_rim_level3', nargs='+', type=int, default=[8])
    # rim_levels_output_dim shouldn't huge (set some thing like 5)
    parser.add_argument('--rim_level1_output_dim', type=int, default=8,
                        help='output size of rim level1')
    parser.add_argument('--rim_level2_output_dim', type=int, default=8,
                        help='output size of rim level2')
    parser.add_argument('--rim_level3_output_dim', type=int, default=8,
                        help='output size of rim level3')

    parser.add_argument('--norm_rim_level1_output', type=boolean_argument, default=False, help='normalise rim level 1 output')

    parser.add_argument('--policy_rim_level1_output_embedding_dim', type=int, default=None)

    parser.add_argument('--rim_level1_condition_on_task_inference_latent', type=boolean_argument, default=True,
                        help='rim level 1 get information from task inference output')
    parser.add_argument('--rim_level2_condition_on_task_inference_latent', type=boolean_argument, default=False,
                        help='rim level 2 get information from task inference output')
    parser.add_argument('--rim_top_down_level3_level2', type=boolean_argument, default=False,
                        help='rim level 2 get information from level 3')
    parser.add_argument('--rim_top_down_level2_level1', type=boolean_argument, default=False,
                        help='rim level 1 get information from level 2')
    # memory
    parser.add_argument('--use_memory', type=boolean_argument, default=False,
                        help='whatever or not use memory in model')
    parser.add_argument('--use_hebb', type=boolean_argument, default=False,
                        help='whatever or not use hebbian memory in memory module')
    parser.add_argument('--use_gen', type=boolean_argument, default=False,
                        help='whatever or not use generative memory in memory module')

    parser.add_argument('--memory_controller_hidden_size', type=int, default=0,
                        help='hidden size of memory controller')
    parser.add_argument('--memory_controller_rim_or_gru', type=str, default='GRU',
                        help='as RNN network for memory controller use GRU of RIM')
    parser.add_argument('--memory_key_dim', type=int, default=0)
    parser.add_argument('--memory_value_dim', type=int, default=0)
    parser.add_argument('--memory_query_dim', type=int, default=0)
    # visoin core
    parser.add_argument('--use_stateless_vision_core', type=boolean_argument, default=False,
                        help='use attentional visual process unit')


    return parser.parse_args(rest_args)
