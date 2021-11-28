"""
Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
"""
import torch.nn as nn
import torch.optim as optim

from utils import helpers as utl


class A2C:
    def __init__(self,
                 args,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 policy_optimiser,
                 policy_anneal_lr,
                 train_steps,
                 optimiser_vae=None,
                 lr=None,
                 eps=None,
                 ):
        self.args = args

        # the model
        self.actor_critic = actor_critic

        # coefficients for mixing the value and entropy loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.use_state_encoder = self.args.policy_state_embedding_dim is not None and self.args.pass_state_to_policy
        self.use_task_inference_latent_encoder = self.args.policy_task_inference_latent_embedding_dim is not None and self.args.pass_task_inference_latent_to_policy
        self.use_belief_encoder = self.args.policy_belief_embedding_dim is not None and self.args.pass_belief_to_policy
        self.use_task_encoder = self.args.policy_task_embedding_dim is not None and self.args.pass_task_to_policy
        self.use_rim_level1_output_encoder = self.args.policy_rim_level1_output_embedding_dim is not None and self.args.use_rim_level1

        # optimiser
        if policy_optimiser == 'adam':
            self.optimiser = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        elif policy_optimiser == 'rmsprop':
            self.optimiser = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=0.99)
        self.optimiser_vae = optimiser_vae

        self.lr_scheduler_policy = None
        self.lr_scheduler_encoder = None
        if policy_anneal_lr:
            lam = lambda f: 1 - f / train_steps
            self.lr_scheduler_policy = optim.lr_scheduler.LambdaLR(self.optimiser, lr_lambda=lam)
            if hasattr(self.args, 'rlloss_through_encoder') and self.args.rlloss_through_encoder:
                self.lr_scheduler_encoder = optim.lr_scheduler.LambdaLR(self.optimiser_vae, lr_lambda=lam)

    def update(self,
               policy_storage,
               encoder=None,  # variBAD encoder
               rlloss_through_encoder=False,  # whether or not to backprop RL loss through encoder
               compute_vae_loss=None,  # function that can compute the VAE loss
               compute_n_step_value_prediction_loss=None,  # function that can compute the n step value prediction loss
               compute_memory_loss=None,
               activated_branch='exploration',
               ):

        # get action values
        advantages = policy_storage.returns[:-1] - policy_storage.value_preds[:-1]

        if rlloss_through_encoder:
            # re-compute encoding (to build the computation graph from scratch)
            utl.recompute_embeddings(self.actor_critic, policy_storage, encoder, sample=False, update_idx=0,
                                     detach_every=self.args.tbptt_stepsize if hasattr(self.args,
                                                                                      'tbptt_stepsize') else None,
                                     activated_branch=activated_branch)

        # update the normalisation parameters of policy inputs before updating
        self.actor_critic.update_rms(args=self.args, policy_storage=policy_storage)

        # call this to make sure that the action_log_probs are computed
        # (needs to be done right here because of some caching thing when normalising actions)
        policy_storage.before_update(self.actor_critic)

        data_generator = policy_storage.feed_forward_generator(advantages, 1)
        for sample in data_generator:

            state_batch, belief_batch, task_batch, \
            actions_batch, latent_sample_batch, latent_mean_batch, latent_logvar_batch, brim_output_level1_batch, brim_output_level2_batch, brim_output_level3_batch, policy_embedded_state_batch,\
            value_preds_batch, return_batch, old_action_log_probs_batch, adv_targ = sample

            if not rlloss_through_encoder:
                state_batch = state_batch.detach()
                if latent_sample_batch is not None:
                    latent_sample_batch = latent_sample_batch.detach()
                    latent_mean_batch = latent_mean_batch.detach()
                    latent_logvar_batch = latent_logvar_batch.detach()

            latent_batch = utl.get_latent_for_policy(sample_embeddings=self.args.sample_embeddings,
                                                         add_nonlinearity_to_latent=self.args.add_nonlinearity_to_latent,
                                                         latent_sample=latent_sample_batch,
                                                         latent_mean=latent_mean_batch,
                                                         latent_logvar=latent_logvar_batch
                                                         )

            values, action_log_probs, dist_entropy = \
                self.actor_critic.evaluate_actions(embedded_state=policy_embedded_state_batch, latent=latent_batch, brim_output_level1=brim_output_level1_batch,
                                                       belief=belief_batch, task=task_batch,
                                                       action=actions_batch, return_action_mean=True
                                                       )

            # --  UPDATE --
            # zero out the gradients
            self.optimiser.zero_grad()
            if rlloss_through_encoder:
                self.optimiser_vae.zero_grad()

            # compute policy loss and backprop
            value_loss = (return_batch - values).pow(2).mean()
            action_loss = -(adv_targ.detach() * action_log_probs).mean()

            # (loss = value loss + action loss + entropy loss, weighted)
            loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef

            # compute vae loss and backprop
            if rlloss_through_encoder:
                loss += self.args.vae_loss_coeff * compute_vae_loss()
                if self.args.use_rim_level2:
                    value_prediction_loss = compute_n_step_value_prediction_loss(self.actor_critic, activated_branch)
                    loss += self.args.n_step_value_prediction_coeff * value_prediction_loss
                if self.args.use_memory and self.args.reconstruction_memory_loss:
                    loss += self.args.reconstruction_memory_loss_coef * compute_memory_loss(self.actor_critic,
                                                                                            activated_branch)

            # compute gradients (will attach to all networks involved in this computation)
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.args.policy_max_grad_norm)
            if encoder is not None and rlloss_through_encoder:
                nn.utils.clip_grad_norm_(encoder.parameters(), self.args.policy_max_grad_norm)

            # update
            self.optimiser.step()
            if rlloss_through_encoder:
                self.optimiser_vae.step()

        if (not rlloss_through_encoder) and (self.optimiser_vae is not None):
            for _ in range(self.args.num_vae_updates):
                compute_vae_loss(update=True)

        if self.lr_scheduler_policy is not None:
            self.lr_scheduler_policy.step()
        if self.lr_scheduler_encoder is not None:
            self.lr_scheduler_encoder.step()

        return value_loss, action_loss, dist_entropy, loss

    def act(self, embedded_state, latent, brim_output_level1, belief, task, deterministic=False):
        return self.actor_critic.act(embedded_state=embedded_state, latent=latent,
                                     brim_output_level1=brim_output_level1, belief=belief, task=task,
                                     deterministic=deterministic)
