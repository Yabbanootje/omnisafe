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
"""Implementation of the Early terminated version of the PPO algorithm."""


from omnisafe.adapter.early_terminated_adapter import EarlyTerminatedAdapter
from omnisafe.adapter.early_terminated_curriculum_adapter import EarlyTerminatedCurriculumAdapter
from omnisafe.adapter.early_terminated_adaptive_curriculum_adapter import EarlyTerminatedAdaptiveCurriculumAdapter
from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.ppo import PPO
from omnisafe.utils import distributed

from omnisafe.algorithms.on_policy.base.policy_gradient import *


@registry.register
class PPOEarlyTerminated(PPO):
    """The Early terminated version of the PPO algorithm.

    A simple combination of the Early terminated RL and the Proximal Policy Optimization algorithm.

    References:
        - Title: Safe Exploration by Solving Early Terminated MDP.
        - Authors: Hao Sun, Ziping Xu, Meng Fang, Zhenghao Peng, Jiadong Guo, Bo Dai, Bolei Zhou.
        - URL: `PPOEarlyTerminated <https://arxiv.org/pdf/2107.04200.pdf>`_
    """

    def _init_env(self) -> None:
        """Initialize the environment.

        OmniSafe uses :class:`omnisafe.adapter.EarlyTerminatedAdapter` to adapt the environment to
        the algorithm.

        User can customize the environment by inheriting this method.

        Examples:
            >>> def _init_env(self) -> None:
            ...     self._env = CustomAdapter()
        """
        if re.search(r"From(\d+|T)HMA(\d+|T)", self._env_id) is not None:
            self._env: EarlyTerminatedAdaptiveCurriculumAdapter = EarlyTerminatedAdaptiveCurriculumAdapter(
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
            )        
        elif re.search(r"From(\d+|T)HMR?(\d+|T)", self._env_id) is not None:
            self._env: EarlyTerminatedCurriculumAdapter = EarlyTerminatedCurriculumAdapter(
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
            )
        else:
            self._env: EarlyTerminatedAdapter = EarlyTerminatedAdapter(
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
            )
        assert (self._cfgs.algo_cfgs.steps_per_epoch) % (
            distributed.world_size() * self._cfgs.train_cfgs.vector_env_nums
        ) == 0, 'The number of steps per epoch is not divisible by the number of environments.'
        self._steps_per_epoch: int = (
            self._cfgs.algo_cfgs.steps_per_epoch
            // distributed.world_size()
            // self._cfgs.train_cfgs.vector_env_nums
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
            self._env: EarlyTerminatedAdaptiveCurriculumAdapter = EarlyTerminatedAdaptiveCurriculumAdapter(
                self._env_id,
                self._cfgs.train_cfgs.vector_env_nums,
                self._seed,
                self._cfgs,
            )
        else:
            self._env: EarlyTerminatedCurriculumAdapter = EarlyTerminatedCurriculumAdapter(
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
