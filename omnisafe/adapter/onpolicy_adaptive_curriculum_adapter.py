from omnisafe.adapter.onpolicy_curriculum_adapter import *
from omnisafe.utils.config import Config
import re

class OnPolicyAdaptiveCurriculumAdapter(OnPolicyCurriculumAdapter):
    def __init__(  # pylint: disable=too-many-arguments
        self,
        env_id: str,
        num_envs: int,
        seed: int,
        cfgs: Config,
    ) -> None:
        """Initialize an instance of :class:`OnPolicyCurriculumAdapter`."""
        super_cfgs = cfgs.copy()
        beta = super_cfgs["env_cfgs"].pop('beta', None)
        kappa = super_cfgs["env_cfgs"].pop('kappa', None)
        early_stop_before = super_cfgs["env_cfgs"].pop('early_stop_before', None)
        super_cfgs = Config.dict2config(super_cfgs)
        super().__init__(env_id, num_envs, seed, super_cfgs)
        print("initialized an OnPolicyAdaptiveCurriculumAdapter with cfgs:", cfgs)
        self._env.set_beta(beta)
        self._env.set_kappa(kappa)

        start_version_pattern = r'From(\d+|T)'
        start_version = re.search(start_version_pattern, env_id)
        if start_version.group(1) == "T":
            self._current_task = 6
        else:
            self._current_task = int(start_version.group(1))
        self.saved_current_task = 0

    def rewrap(self, model_params) -> None:
        """Rewrap the environment with the given normalizers.

        Args:
            model_params (Any): torch object containing parameters for the relevant normalizers.
        """
        env_cfgs = {}

        if hasattr(self._cfgs, 'env_cfgs') and self._cfgs.env_cfgs is not None:
            env_cfgs = self._cfgs.env_cfgs.todict()

        beta = env_cfgs.pop('beta', None)
        kappa = env_cfgs.pop('kappa', None)

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

        obs, info = self.reset()
        if info['current_task'] > self._current_task:
            self.saved_current_task = 0
        self._current_task = info['current_task']
        start_obs = obs.clone().detach()
        done_task = False
        time_out_task = False
        self._tasks_done = 0

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
                # done represents if the goal state has been reached
                # epoch_end represents if the amount of steps exceeds steps_per_epoch
                # time_out represents if a single episode exceeds steps_per_epoch
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
                        self._tasks_done += 1

                    if done:
                        done_task = True

                    if time_out:
                        time_out_task = True

                    buffer.finish_path(last_value_r, last_value_c, idx)


        action, value_r, value_c, log_prob = agent.forward(start_obs)
        # print(f"The observation {start_obs}\n leads to action: {action},\n value_r: {value_r}, \n value_c: {value_c} and \n log_prob: {log_prob}")
        print("In OnPolicyAdaptiveCurriculumAdapter we have:", type(self._env).__name__)
        print(f"The task at current epoch is{' not' if not done_task else ''} completed and according to the logger the costs are {logger.get_stats('Metrics/EpCost')[0]}")
        metric_dict = {
            "action": action,
            "value_r": value_r,
            "value_c": value_c,
            "log_prob": log_prob,
            "start_obs": start_obs,
            "done_task": done_task,
            "time_out_task": time_out_task,
            "Metrics/EpRet": logger.get_stats("Metrics/EpRet")[0],
            "Metrics/EpCost": logger.get_stats("Metrics/EpCost")[0],
            "Metrics/EpLen": logger.get_stats("Metrics/EpLen")[0],
            'Completed_episodes': self._tasks_done,
            "Value/reward": logger.get_stats("Value/reward")[0],
            "Value/cost": logger.get_stats("Value/cost")[0],
        }
        ready_for_next_task = self._env.update(metric_dict)
        if self.saved_current_task < 1:
            logger.store({"Ready_for_next_task": int(ready_for_next_task)})
            if ready_for_next_task:
                self.saved_current_task += 1
        else:
            logger.store({"Ready_for_next_task": 0})