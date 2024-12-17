# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaacgym import gymtorch

from pm.envs.pm_inversion.base_task.common import BaseMaskedMimicTask

from pm.envs.pm_inversion.masked_mimic.masked_mimic_humanoid import (
    MaskedMimicHumanoid,
)


class MaskedMimicTaskHumanoid(BaseMaskedMimicTask, MaskedMimicHumanoid):  # type: ignore[misc]
    def __init__(self, config, device, motion_lib=None):
        config.visualize_markers = False
        super().__init__(config=config, device=device, motion_lib=motion_lib)

    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)
        self.build_env_task(env_id, env_ptr, humanoid_asset)

    def build_env_task(self, env_id, env_ptr, humanoid_asset):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_env_tensors(self, env_ids):
        super().reset_env_tensors(env_ids)

        env_ids_int32 = self.get_task_actor_ids_for_reset(env_ids)
        if env_ids_int32 is not None:
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_states),
                gymtorch.unwrap_tensor(env_ids_int32),
                len(env_ids_int32),
            )

    def pre_physics_step(self, actions):
        self.update_task(actions)
        super().pre_physics_step(actions)

    def update_task(self, actions):
        pass

    ###############################################################
    # Getters
    ###############################################################
    def get_task_actor_ids_for_reset(self, env_ids):
        return None

    ###############################################################
    # Helpers
    ###############################################################
    def render(self):
        super().render()

        if self.viewer:
            # print reward
            print(f"Reward: {self.rew_buf.item():.2f}")
            self.draw_task()
