from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import MaskedMimicTaskHumanoid


class PMWrapper:
    def __init__(self, env: MaskedMimicTaskHumanoid):
        self.env = env
        self.reward_raw = env.rew_buf

    # Delegate all other attribute/method calls to the wrapped environment
    def __getattr__(self, name):
        return getattr(self.env, name)

    def _create_envs(self, num_envs, spacing, num_per_row):
        return self.env.create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        return self.env.build_env(env_id, env_ptr, humanoid_asset)

    def _update_task(self):
        return self.env.update_task()

    def _reset_task(self, env_ids):
        return self.env.reset_task(env_ids)

    def _compute_task_obs(self, env_ids=None):
        return self.env.compute_task_obs(env_ids)

    def _compute_reward(self, actions):
        return self.env.compute_reward(actions)

    def _draw_task(self):
        return self.env.draw_task()
