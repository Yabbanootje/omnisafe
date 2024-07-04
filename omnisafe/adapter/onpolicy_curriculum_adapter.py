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
"""OnPolicy Adapter for OmniSafe."""

from __future__ import annotations

from typing import Any

import torch
from rich.progress import track

from omnisafe.adapter.online_adapter import OnlineAdapter
from omnisafe.adapter.onpolicy_adapter import *
from omnisafe.adapter.online_adapter import *
from omnisafe.common.buffer import VectorOnPolicyBuffer
from omnisafe.common.logger import Logger
from omnisafe.models.actor_critic.constraint_actor_critic import ConstraintActorCritic
from omnisafe.utils.config import Config
from omnisafe.common import Normalizer


class OnPolicyCurriculumAdapter(OnPolicyAdapter):
    """OnPolicy Adapter for OmniSafe.

    :class:`OnPolicyAdapter` is used to adapt the environment to the on-policy training.

    Args:
        env_id (str): The environment id.
        num_envs (int): The number of environments.
        seed (int): The random seed.
        cfgs (Config): The configuration.
    """

    _ep_ret: torch.Tensor
    _ep_cost: torch.Tensor
    _ep_len: torch.Tensor

    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        self._num_envs: int = num_envs
        self._seed: int = seed

    def rewrap(self, model_params) -> None:
        """Rewrap the environment with the given normalizers.

        Args:
            model_params (Any): torch object containing parameters for the relevant normalizers.
        """
        env_cfgs = {}

        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()

        self._env: CMDP = make(self._env_id, num_envs=self._num_envs, device=self._device, **env_cfgs)

        if self._env.need_time_limit_wrapper:
            assert (
                self._env.max_episode_steps
            ), 'You must define max_episode_steps as an integer\
                \nor cancel the use of the time_limit wrapper.'
            self._env = TimeLimit(
                self._env,
                time_limit=self._env.max_episode_steps,
                device=self._device,
            )
        if self._env.need_auto_reset_wrapper:
            self._env = AutoReset(self._env, device=self._device)
        if self._cfgs['algo_cfgs']['obs_normalize']:
            obs_normalizer = Normalizer(shape=self.observation_space.shape, clip=5)
            obs_normalizer.load_state_dict(model_params['obs_normalizer'])
            self._env = ObsNormalize(self._env, device=self._device, norm=obs_normalizer)
        if self._cfgs['algo_cfgs']['reward_normalize']:
            reward_normalize = Normalizer(shape=(), clip=5)
            reward_normalize.load_state_dict(model_params['reward_normalize'])
            self._env = RewardNormalize(self._env, device=self._device, norm=reward_normalize)
        if self._cfgs['algo_cfgs']['cost_normalize']:
            cost_normalize = Normalizer(shape=(), clip=5)
            cost_normalize.load_state_dict(model_params['cost_normalize'])
            self._env = CostNormalize(self._env, device=self._device, norm=cost_normalize)
        if self._env.num_envs == 1:
            self._env = Unsqueeze(self._env, device=self._device)

        self._eval_env: CMDP | None = None
        if self._env.need_evaluation:
            self._eval_env = make(self._env_id, num_envs=1, device=self._device, **env_cfgs)

            assert self._eval_env, 'Your environment for evaluation does not exist!'
            if self._env.need_time_limit_wrapper:
                assert (
                    self._eval_env.max_episode_steps
                ), 'You must define max_episode_steps as an\
                    \ninteger or cancel the use of the time_limit wrapper.'
                self._eval_env = TimeLimit(
                    self._eval_env,
                    time_limit=self._eval_env.max_episode_steps,
                    device=self._device,
                )
            if self._env.need_auto_reset_wrapper:
                self._eval_env = AutoReset(self._eval_env, device=self._device)
            if self._cfgs['algo_cfgs']['obs_normalize']:
                self._eval_env = ObsNormalize(self._eval_env, device=self._device, norm=obs_normalizer)
            self._eval_env = ActionScale(self._eval_env, low=-1.0, high=1.0, device=self._device)
            self._eval_env = Unsqueeze(self._eval_env, device=self._device)

        self._env.set_seed(self._seed)


    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ) -> None:
        """Rollout the environment and store the data in the buffer.

        .. warning::
            As OmniSafe uses :class:`AutoReset` wrapper, the environment will be reset automatically,
            so the final observation will be stored in ``info['final_observation']``.

        Args:
            steps_per_epoch (int): Number of steps per epoch.
            agent (ConstraintActorCritic): Constraint actor-critic, including actor , reward critic
                and cost critic.
            buffer (VectorOnPolicyBuffer): Vector on-policy buffer.
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
        """
        self._reset_log()

        # disable = False
        # if hasattr(self._env, "disable_progress"):
        #     disable = self._env.disable_progress

        obs, info = self.reset()
        self._current_task = info['current_task']
        
        for step in track(
            range(steps_per_epoch),
            description=f'Processing rollout for epoch: {logger.current_epoch}...',
            # disable=disable
        ):
            act, value_r, value_c, logp = agent.step(obs)
            next_obs, reward, cost, terminated, truncated, info = self.step(act)

            self._log_value(reward=reward, cost=cost, info=info)

            if self._cfgs.algo_cfgs.use_cost:
                logger.store({'Value/cost': value_c})
            logger.store({'Value/reward': value_r})

            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                logp=logp,
            )

            obs = next_obs
            epoch_end = step >= steps_per_epoch - 1
            if epoch_end:
                num_dones = int(terminated.contiguous().sum())
                if self._env.num_envs - num_dones:
                    logger.log(
                        f'\nWarning: trajectory cut off when rollout by epoch\
                            in {self._env.num_envs - num_dones} of {self._env.num_envs} environments.',
                    )

            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            _, last_value_r, last_value_c, _ = agent.step(obs[idx])
                        if time_out:
                            _, last_value_r, last_value_c, _ = agent.step(
                                info['final_observation'][idx],
                            )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)

                    if done or time_out:
                        self._log_metrics(logger, idx)
                        self._reset_log(idx)

                        self._ep_ret[idx] = 0.0
                        self._ep_cost[idx] = 0.0
                        self._ep_len[idx] = 0.0

                    buffer.finish_path(last_value_r, last_value_c, idx)

    def _log_metrics(self, logger: Logger, idx: int) -> None:
        """Log metrics, including ``EpRet``, ``EpCost``, ``EpLen``.

        Args:
            logger (Logger): Logger, to log ``EpRet``, ``EpCost``, ``EpLen``.
            idx (int): The index of the environment.
        """
        if hasattr(self._env, 'spec_log'):
            self._env.spec_log(logger)
        logger.store(
            {
                'Metrics/EpRet': self._ep_ret[idx],
                'Metrics/EpCost': self._ep_cost[idx],
                'Metrics/EpLen': self._ep_len[idx],
                'Current_task': self._current_task,
            },
        )
