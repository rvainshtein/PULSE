# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import Tuple, Dict

import isaacgym.torch_utils as itu
import numpy as np
import torch
from torch import Tensor
from utils import torch_utils

from phc.env.tasks.pm.direction import HumanoidDirection, compute_heading_reward
from phc.utils.torch_utils_pm import calc_heading_quat
from poselib.poselib.core import rotation3d as rotations

TAR_ACTOR_ID = 1


class HumanoidDirectionFacing(HumanoidDirection):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.config = cfg.env
        self.use_current_pose_obs = self.config.steering_params.get("use_current_pose_obs", False)
        self.pose_obs_size = 6 if self.use_current_pose_obs else 0  # 2 for root and head height, 6 for root and head coords, self.get_obs_size() for full humanoid pose

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)
        self.obs_size = self.get_pm_obs_size()

        self._tar_facing_dir_theta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._tar_facing_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_facing_dir[..., 0] = 1.0

        self._heading_turn_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )

    def get_pm_obs_size(self):
        return super().get_pm_obs_size() + 2

    def _compute_task_obs(self, env_ids=None):
        super()._compute_task_obs(env_ids)
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        root_states = self._root_states[env_ids]
        facing_obs = compute_facing_observations(root_states, self._tar_dir[env_ids])
        obs = torch.cat([self.direction_obs[env_ids], facing_obs], dim=-1)
        return obs
    def reset_heading_task(self, env_ids):
        super().reset_heading_task(env_ids)
        n = len(env_ids)
        if np.random.binomial(1, self._random_heading_probability):
            face_dir_theta = 2 * torch.pi * torch.rand(n, device=self.device) - torch.pi
        else:
            dir_delta_theta = (
                    2 * self._standard_heading_change * torch.rand(n, device=self.device)
                    - self._standard_heading_change
            )
            # map tar_dir_theta back to [0, 2pi], add delta, project back into [0, 2pi] and then shift.
            face_dir_theta = (
                                     dir_delta_theta + self._tar_facing_dir_theta[env_ids] + np.pi
                             ) % (2 * np.pi) - np.pi

        face_tar_dir = torch.stack(
            [torch.cos(face_dir_theta), torch.sin(face_dir_theta)], dim=-1
        )
        self._tar_facing_dir[env_ids] = face_tar_dir
        self._tar_facing_dir_theta[env_ids] = face_dir_theta

        self._heading_turn_steps[env_ids] = (
                30 * 1 + self.progress_buf[env_ids]
        )  # Allow 15 frames (0.5sec) to turn.

    def _compute_reward(self, actions):
        root_states = self._root_states
        root_pos = root_states[..., :3]
        root_rot = root_states[:, 3:7]
        self.rew_buf[:], output_dict = compute_facing_reward(root_pos, self._prev_root_pos, root_rot, self._tar_dir,
                                                             self._tar_speed, self._tar_facing_dir, self.dt)
        self._prev_root_pos[:] = root_pos

        # print the target speed of the env and the speed actually achieved in that direction

        if (
                self.config.num_envs == 1
                and self.config.steering_params.log_speed
                and self.progress_buf % 3 == 0
        ):
            print(
                f'speed: {output_dict["tar_dir_speed"].item():.3f}/{self._tar_speed.item():.3f}'
            )
            print(
                f'error: {output_dict["tar_vel_err"].item():.3f}; tangent error: {output_dict["tangent_vel_err"].item():.3f}'
            )

class HumanoidDirectionFacingZ(HumanoidDirectionFacing):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type,
                         device_id=device_id, headless=headless)
        self.initialize_z_models()
        return

    def step(self, actions):
        self.step_z(actions)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        super()._setup_character_props_z()

        return


#####################################################################
###=========================jit functions=========================###
#####################################################################
@torch.jit.script
def compute_facing_observations(root_states, tar_face_dir):
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    tar_face_dir3d = torch.cat(
        [tar_face_dir, torch.zeros_like(tar_face_dir[..., 0:1])], dim=-1
    )
    local_tar_face_dir = rotations.quat_rotate(heading_rot, tar_face_dir3d)
    local_tar_face_dir = local_tar_face_dir[..., 0:2]
    return local_tar_face_dir


@torch.jit.script
def compute_facing_reward(root_pos: Tensor, prev_root_pos: Tensor, root_rot: Tensor, tar_dir: Tensor, tar_speed: Tensor,
                          tar_face_dir: Tensor, dt: float) -> Tuple[Tensor, Dict[str, Tensor]]:
    dir_reward, output_dict = compute_heading_reward(
        root_pos, prev_root_pos, tar_dir, tar_speed, dt
    )

    dir_reward_w = 0.7
    facing_reward_w = 0.3
    heading_rot = calc_heading_quat(root_rot, True)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = itu.quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    reward = dir_reward_w * dir_reward + facing_reward_w * facing_reward

    output_dict["facing_dir"] = facing_dir
    output_dict["tar_face_dir"] = tar_face_dir
    output_dict["facing_err"] = facing_err
    output_dict["facing_reward"] = facing_reward

    return reward, output_dict
