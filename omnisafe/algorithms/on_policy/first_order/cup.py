# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of the CUP algorithm."""

import torch
from rich.progress import track
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange
from omnisafe.utils import distributed

from omnisafe.algorithms.on_policy.base.policy_gradient import *

@registry.register
class CUP(PPO):
    """The Constrained Update Projection (CUP) Approach to Safe Policy Optimization.

    References:
        - Title: Constrained Update Projection Approach to Safe Policy Optimization
        - Authors: Long Yang, Jiaming Ji, Juntao Dai, Linrui Zhang, Binbin Zhou, Pengfei Li,
            Yaodong Yang, Gang Pan.
        - URL: `CUP <https://arxiv.org/abs/2209.07089>`_
    """

    _p_dist: Normal

    def _init(self) -> None:
        """The initialization of the algorithm.

        Here we additionally initialize the Lagrange multiplier.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the CUP specific information.

        +-----------------------------+----------------------------------------------------------+
        | Things to log               | Description                                              |
        +=============================+==========================================================+
        | Metrics/LagrangeMultiplier  | The Lagrange multiplier.                                 |
        +-----------------------------+----------------------------------------------------------+
        | Loss/Loss_pi_c              | The loss of the cost performance.                        |
        +-----------------------------+----------------------------------------------------------+
        | Train/SecondStepStopIter    | The number of iterations to stop the second step.        |
        +-----------------------------+----------------------------------------------------------+
        | Train/SecondStepEntropy     | The entropy of the current policy.                       |
        +-----------------------------+----------------------------------------------------------+
        | Train/SecondStepPolicyRatio | The ratio between the current policy and the old policy. |
        +-----------------------------+----------------------------------------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier')
        self._logger.register_key('Loss/Loss_pi_c', delta=True)
        self._logger.register_key('Train/SecondStepStopIter')
        self._logger.register_key('Train/SecondStepEntropy')
        self._logger.register_key('Train/SecondStepPolicyRatio', min_and_max=True)

    def _loss_pi_cost(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,
        adv_c: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the performance of cost on this moment.

        We compute the KL divergence between the current policy and the old policy, the entropy of
        the current policy, and the ratio between the current policy and the old policy.

        The loss of the cost performance is defined as:

        .. math::

            L = \underset{a \sim \pi_{\theta}}{\mathbb{E}} [
                \lambda \frac{1 - \gamma \nu}{1 - \gamma}
                    \frac{\pi_{\theta}^{'} (a|s)}{\pi_{\theta} (a|s)} A^{C}_{\pi_{\theta}}
                + KL (\pi_{\theta}^{'} (a|s) || \pi_{\theta} (a|s))
            ]

        where :math:`\lambda` is the Lagrange multiplier, :math:`\frac{1 - \gamma \nu}{1 - \gamma}`
        is the coefficient value, :math:`\pi_{\theta}^{'} (a_t|s_t)` is the current policy,
        :math:`\pi_{\theta} (a_t|s_t)` is the old policy, :math:`A^{C}_{\pi_{\theta}}` is the cost
        advantage, :math:`KL (\pi_{\theta}^{'} (a_t|s_t) || \pi_{\theta} (a_t|s_t))` is the KL
        divergence between the current policy and the old policy.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The loss of the cost performance.
        """
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        ratio = torch.exp(logp_ - logp)

        kl = torch.distributions.kl_divergence(distribution, self._p_dist).sum(-1, keepdim=True)

        coef = (1 - self._cfgs.algo_cfgs.gamma * self._cfgs.algo_cfgs.lam) / (
            1 - self._cfgs.algo_cfgs.gamma
        )
        loss = (self._lagrange.lagrangian_multiplier * coef * ratio * adv_c + kl).mean()

        entropy = distribution.entropy().mean().item()
        self._logger.store(
            {
                'Loss/Loss_pi_c': loss.item(),
                'Train/SecondStepEntropy': entropy,
                'Train/SecondStepPolicyRatio': ratio,
            },
        )
        return loss

    def _update(self) -> None:
        r"""Update actor, critic, and Lagrange multiplier parameters.

        In CUP, the Lagrange multiplier is updated as the naive lagrange multiplier update.

        Then in each iteration of the policy update, CUP calculates current policy's distribution,
        which used to calculate the policy loss.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)

        super()._update()

        data = self._buf.get()
        obs, act, logp, adv_c = (
            data['obs'],
            data['act'],
            data['logp'],
            data['adv_c'],
        )
        original_obs = obs
        with torch.no_grad():
            old_distribution = self._actor_critic.actor(obs)
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

        dataloader = DataLoader(
            dataset=TensorDataset(obs, act, logp, adv_c, old_mean, old_std),
            batch_size=self._cfgs.algo_cfgs.batch_size,
            shuffle=True,
        )

        final_steps = self._cfgs.algo_cfgs.update_iters
        for i in track(range(self._cfgs.algo_cfgs.update_iters), description='Updating...'):
            for obs, act, logp, adv_c, old_mean, old_std in dataloader:
                self._p_dist = Normal(old_mean, old_std)
                loss_cost = self._loss_pi_cost(obs, act, logp, adv_c)
                self._actor_critic.actor_optimizer.zero_grad()
                loss_cost.backward()
                if self._cfgs.algo_cfgs.max_grad_norm is not None:
                    clip_grad_norm_(
                        self._actor_critic.actor.parameters(),
                        self._cfgs.algo_cfgs.max_grad_norm,
                    )
                distributed.avg_grads(self._actor_critic.actor)
                self._actor_critic.actor_optimizer.step()

            new_distribution = self._actor_critic.actor(original_obs)

            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
            )
            kl = distributed.dist_avg(kl)

            if self._cfgs.algo_cfgs.kl_early_stop and kl.item() > self._cfgs.algo_cfgs.target_kl:
                final_steps = i + 1
                self._logger.log(f'Early stopping at iter {i + 1} due to reaching max kl')
                break

        self._logger.store(
            {
                'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier.item(),
                'Train/SecondStepStopIter': final_steps,  # pylint: disable=undefined-loop-variable
            },
        )

    def __load_model_and_env(
        self,
        epoch: int,
        path: str,
    ) -> None:
        """Load the model from the save directory.

        Args:
            epoch (int): The epoch that should be loaded
            path (str): The path to where the model is saved.

        Raises:
            FileNotFoundError: If the model is not found.
        """
        # load the saved model
        model_path = os.path.join(path, 'torch_save', f'epoch-{epoch}.pt')
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error
        
        # Redo _init_env with loaded parameters       
        if re.search(r"From(\d+|T)HMA(\d+|T)", self._env_id) is not None:
            self._env: OnPolicyAdaptiveCurriculumAdapter = OnPolicyAdaptiveCurriculumAdapter(
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
            )
        else:
            self._env: OnPolicyCurriculumAdapter = OnPolicyCurriculumAdapter(
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
            )
            self._env.rewrap(model_params)
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
        )

        # Redo _init_model with loaded parameters
        self._actor_critic: ConstraintActorCritic = ConstraintActorCritic(
            obs_space=self._env.observation_space,
            act_space=self._env.action_space,
            model_cfgs=self._cfgs.model_cfgs,
            epochs=self._cfgs.train_cfgs.epochs,
        ).to(self._device)
        self._actor_critic.load_state_dict(model_params['actor_critic'])
        
        if distributed.world_size() > 1:
            distributed.sync_params(self._actor_critic)

        if self._cfgs.model_cfgs.exploration_noise_anneal:
            self._actor_critic.set_annealing(
                epochs=[0, self._cfgs.train_cfgs.epochs],
                std=self._cfgs.model_cfgs.std_range,
            )

        self._init()

        # Redo part of _init_log
        what_to_save: dict[str, Any] = {}
        what_to_save['pi'] = self._actor_critic.actor
        what_to_save['actor_critic'] = self._actor_critic
        if self._cfgs.algo_cfgs.obs_normalize:
            obs_normalizer = self._env.save()['obs_normalizer']
            what_to_save['obs_normalizer'] = obs_normalizer
        if self._cfgs.algo_cfgs.reward_normalize:
            reward_normalizer = self._env.save()['reward_normalizer']
            what_to_save['reward_normalizer'] = reward_normalizer
        if self._cfgs.algo_cfgs.cost_normalize:
            cost_normalizer = self._env.save()['cost_normalizer']
            what_to_save['cost_normalizer'] = cost_normalizer
        self._logger.setup_torch_saver(what_to_save)
        self._logger.torch_save()

    def __get_adaptive_epoch(self,
        path: str
    ) -> int:
        csv_df = pd.read_csv(path)
        last_task = csv_df["Current_task"].iloc[-1]
        last_task_df = csv_df[csv_df['Current_task'] == last_task]
        for _, row in last_task_df.iterrows():
            if row["Ready_for_next_task"] == 1:
                return int(row["Train/Epoch"] + 1)
        # If the previous save did not reach the point of being ready for the next task,
        # return the last row with the idea that this is also the last epoch for this new iteration,
        # meaning that it will immediately stop
        return int(last_task_df["Train/Epoch"].iloc[-1] + 1)

    def load(self, 
        path: str,
        epoch: int | None = None, 
    ) -> None:
        """Load a saved model.

        Args:
            path (str): The path to the saved model that should be loaded.
            epoch (int or None): The epoch to be loaded.
        """
        if epoch is None:
            epoch = self.__get_adaptive_epoch(os.path.join(path, "progress.csv"))
            
        self.__load_model_and_env(epoch=epoch, path=path)

        self._start_epoch = epoch
        self._logger.set_current_epoch(epoch)

        self._logger.copy_from_csv(os.path.join(path, "progress.csv"))

        # Keep the lagrangian multiplier
        csv_df = pd.read_csv(os.path.join(path, "progress.csv"))
        final_epoch_df = csv_df[csv_df["Train/Epoch"] == epoch - 1]
        self._cfgs.lagrange_cfgs["lagrangian_multiplier_init"] = final_epoch_df.iloc[0]["Metrics/LagrangeMultiplier"]
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)
