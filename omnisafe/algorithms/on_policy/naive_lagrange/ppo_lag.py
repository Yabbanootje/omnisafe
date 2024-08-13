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
"""Implementation of the Lagrange version of the PPO algorithm."""

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.common.lagrange import Lagrange

import pandas as pd
from omnisafe.algorithms.on_policy.base.policy_gradient import *

@registry.register
class PPOLag(PPO):
    """The Lagrange version of the PPO algorithm.

    A simple combination of the Lagrange method and the Proximal Policy Optimization algorithm.
    """

    def _init(self) -> None:
        """Initialize the PPOLag specific model.

        The PPOLag algorithm uses a Lagrange multiplier to balance the cost and reward.
        """
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Log the PPOLag specific information.

        +----------------------------+--------------------------+
        | Things to log              | Description              |
        +============================+==========================+
        | Metrics/LagrangeMultiplier | The Lagrange multiplier. |
        +----------------------------+--------------------------+
        """
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        r"""Update actor, critic, as we used in the :class:`PolicyGradient` algorithm.

        Additionally, we update the Lagrange multiplier parameter by calling the
        :meth:`update_lagrange_multiplier` method.

        .. note::
            The :meth:`_loss_pi` is defined in the :class:`PolicyGradient` algorithm. When a
            lagrange multiplier is used, the :meth:`_loss_pi` method will return the loss of the
            policy as:

            .. math::

                L_{\pi} = -\underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                    \frac{\pi_{\theta} (a_t|s_t)}{\pi_{\theta}^{old}(a_t|s_t)}
                    [ A^{R}_{\pi_{\theta}} (s_t, a_t) - \lambda A^{C}_{\pi_{\theta}} (s_t, a_t) ]
                \right]

            where :math:`\lambda` is the Lagrange multiplier parameter.
        """
        # note that logger already uses MPI statistics across all processes..
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        # first update Lagrange multiplier parameter
        self._lagrange.update_lagrange_multiplier(Jc)
        # then update the policy and value function
        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Compute surrogate loss.

        PPOLag uses the following surrogate loss:

        .. math::

            L = \frac{1}{1 + \lambda} [
                A^{R}_{\pi_{\theta}} (s, a)
                - \lambda A^C_{\pi_{\theta}} (s, a)
            ]

        Args:
            adv_r (torch.Tensor): The ``reward_advantage`` sampled from buffer.
            adv_c (torch.Tensor): The ``cost_advantage`` sampled from buffer.

        Returns:
            The advantage function combined with reward and cost.
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1 + penalty)
    
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
            epoch (int): The epoch to be loaded.
            path (str): The path to the saved model that should be loaded.
        """
        if epoch is None:
            epoch = self.__get_adaptive_epoch(os.path.join(path, "progress.csv"))
            
        self.__load_model_and_env(epoch=epoch, path=path)

        self._start_epoch = epoch
        self._logger.set_current_epoch(epoch)

        self._logger.copy_from_csv(os.path.join(path, "progress.csv"))

        csv_df = pd.read_csv(os.path.join(path, "progress.csv"))
        final_epoch_df = csv_df[csv_df["Train/Epoch"] == epoch - 1]
        self._cfgs.lagrange_cfgs["lagrangian_multiplier_init"] = final_epoch_df.iloc[0]["Metrics/LagrangeMultiplier"]
        self._lagrange = Lagrange(**self._cfgs.lagrange_cfgs)