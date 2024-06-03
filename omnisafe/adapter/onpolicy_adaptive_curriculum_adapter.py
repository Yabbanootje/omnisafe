from omnisafe.adapter.onpolicy_adapter import *
from omnisafe.utils.config import Config

class OnPolicyAdaptiveCurriculumAdapter(OnPolicyAdapter):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyCurriculumAdapter`."""
        super().__init__(env_id, num_envs, seed, cfgs)
        print("initialized an OnPolicyAdaptiveCurriculumAdapter")

    def rollout(  # pylint: disable=too-many-locals
        self,
        steps_per_epoch: int,
        agent: ConstraintActorCritic,
        buffer: VectorOnPolicyBuffer,
        logger: Logger,
    ):
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

        obs, _ = self.reset()
        start_obs = obs.clone().detach()
        completed_task = False

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
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1)
                    last_value_c = torch.zeros(1)
                    if not done:
                        if epoch_end:
                            logger.log(
                                f'Warning: trajectory cut off when rollout by epoch at {self._ep_len[idx]} steps.',
                            )
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

                    if done:
                        completed_task = True

                    buffer.finish_path(last_value_r, last_value_c, idx)


        # action, value_r, value_c, log_prob = agent.forward(start_obs)
        # print(f"The observation {start_obs}\n leads to action: {action},\n value_r: {value_r}, \n value_c: {value_c} and \n log_prob: {log_prob}")
        print("In OnPolicyAdaptiveCurriculumAdapter we have:", type(self._env).__name__)
        print(f"The task at current epoch is {'not' if not completed_task else ''} completed and according to the logger the costs are {logger.get_stats('Metrics/EpCost')[0]}")
        self._env.update((completed_task, logger.get_stats("Metrics/EpCost")[0]))