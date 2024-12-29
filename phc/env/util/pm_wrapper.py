from phys_anim.envs.masked_mimic_inversion.base_task.isaacgym import MaskedMimicTaskHumanoid


class PMWrapper:
    def __init__(self, env: MaskedMimicTaskHumanoid):
        self.env = env
        self.attribute_mapping = {
            'reward_raw': 'rew_buf',
            'num_actions': 'num_act',
            # 'num_states': 'num_joints'  # not sure what this is and why we use it even.
            'num_states': 'return_0',  # not sure what this is and why we use it even.
            'get_num_amp_obs': 'get_num_disc_obs',
            'get_num_enc_amp_obs': 'get_num_disc_obs',
        }

    # Delegate all other attribute/method calls to the wrapped environment
    def __getattr__(self, name):
        if name in self.attribute_mapping:
            # Map wrapper attribute to the corresponding env attribute
            new_name = self.attribute_mapping[name]
            if new_name == 'return_0':
                return 0
            else:
                return getattr(self.env, new_name)
        else:
            # Fallback to normal delegation
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
