# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from torch import Tensor
from typing import TYPE_CHECKING, Optional, Tuple, Dict

from pm.isaac_utils import rotations, torch_utils
import torch

if TYPE_CHECKING:
    from pm.envs.pm_inversion.steering.isaacgym import (
        MaskedMimicDirectionHumanoid,
    )
else:
    MaskedMimicDirectionHumanoid = object


class MaskedMimicBaseDirection(MaskedMimicDirectionHumanoid):  # type: ignore[misc]
    def __init__(self, config, device: torch.device):
        super().__init__(config=config, device=device)

        self._tar_speed_min = self.config.steering_params.tar_speed_min
        self._tar_speed_max = self.config.steering_params.tar_speed_max

        self._heading_change_steps_min = self.config.steering_params.change_steps_min
        self._heading_change_steps_max = self.config.steering_params.change_steps_max
        self._random_heading_probability = self.config.steering_params.random_heading_probability
        self._standard_heading_change = self.config.steering_params.standard_heading_change
        self._standard_speed_change = self.config.steering_params.standard_speed_change
        self._stop_probability = self.config.steering_params.stop_probability
        self.use_current_pose_obs = config.steering_params.get("use_current_pose_obs", False)
        if "smpl" in self.config.robot.asset.asset_file_name:
            self.head_id = self.get_body_id("Head")

        if self.use_current_pose_obs:
            self.direction_obs = torch.zeros(
                (
                    config.num_envs,
                    config.steering_params.obs_size + self.get_obs_size(),
                ),
                device=device,
                dtype=torch.float,
            )
        else:
            self.head_id = self.get_body_id("head")

        self.pose_obs_size = 6 if self.use_current_pose_obs else 0  # 2 for root and head height, 6 for root and head coords, self.get_obs_size() for full humanoid pose

        self.direction_obs = torch.zeros(
            (config.num_envs, config.steering_params.obs_size + self.pose_obs_size),  # 6 for root and head coords
            device=device,
            dtype=torch.float,
        )

        self.inversion_obs = self.direction_obs

        self._heading_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._tar_dir_theta = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        self._tar_dir = torch.zeros(
            [self.num_envs, 2], device=self.device, dtype=torch.float
        )
        self._tar_dir[..., 0] = 1.0

        self._tar_speed = torch.ones(
            [self.num_envs], device=self.device, dtype=torch.float
        )

    def reset_task(self, env_ids):
        if len(env_ids) > 0:
            self.reset_heading_task(env_ids)
        super().reset_task(env_ids)

    def update_task(self, actions):
        super().update_task(actions)

        reset_task_mask = self.progress_buf >= self._heading_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self.reset_heading_task(rest_env_ids)

    def reset_heading_task(self, env_ids):
        n = len(env_ids)
        if np.random.binomial(1, self._random_heading_probability):
            dir_theta = 2 * np.pi * torch.rand(n, device=self.device) - np.pi
            tar_speed = (self._tar_speed_max - self._tar_speed_min) * torch.rand(
                n, device=self.device
            ) + self._tar_speed_min
        else:
            dir_delta_theta = (
                    2 * self._standard_heading_change * torch.rand(n, device=self.device)
                    - self._standard_heading_change
            )
            # map tar_dir_theta back to [0, 2pi], add delta, project back into [0, 2pi] and then shift.
            dir_theta = (dir_delta_theta + self._tar_dir_theta[env_ids] + np.pi) % (
                    2 * np.pi
            ) - np.pi

            speed_delta = (
                    2 * self._standard_speed_change * torch.rand(n, device=self.device)
                    - self._standard_speed_change
            )
            tar_speed = torch.clamp(
                speed_delta + self._tar_speed[env_ids],
                min=self._tar_speed_min,
                max=self._tar_speed_max,
            )

        tar_dir = torch.stack([torch.cos(dir_theta), torch.sin(dir_theta)], dim=-1)

        change_steps = torch.randint(
            low=self._heading_change_steps_min,
            high=self._heading_change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )

        stop_probs = torch.ones(n, device=self.device) * self._stop_probability
        should_stop = torch.bernoulli(stop_probs)

        self._tar_speed[env_ids] = tar_speed * (1.0 - should_stop)
        self._tar_dir_theta[env_ids] = dir_theta
        self._tar_dir[env_ids] = tar_dir
        self._heading_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps

    def compute_observations(self, env_ids=None):
        self.mask_everything()
        super().compute_observations(env_ids)
        self.mask_everything()

    def compute_task_obs(self, env_ids=None):
        super().compute_task_obs(env_ids)

        if env_ids is None:
            root_states = self.get_humanoid_root_states()
            tar_dir = self._tar_dir
            tar_speed = self._tar_speed
            # humanoid_obs = self.obs_buf
            global_translations = self.get_body_positions()
            # root_height = global_translations[:, 0, 2]
            # head_height = global_translations[:, self.head_id, 2]
            # humanoid_obs = torch.cat([root_height.unsqueeze(-1), head_height.unsqueeze(-1)], dim=-1)
            root_coords = global_translations[:, 0, :]
            head_coords = global_translations[:, self.head_id, :]
            humanoid_obs = torch.cat([root_coords, head_coords], dim=-1)
        else:
            root_states = self.get_humanoid_root_states()[env_ids]
            tar_dir = self._tar_dir[env_ids]
            tar_speed = self._tar_speed[env_ids]
            # humanoid_obs = self.obs_buf[env_ids]
            global_translations = self.get_body_positions()[env_ids]
            # root_height = global_translations[env_ids, 0, 2]
            # head_height = global_translations[env_ids, self.head_id, 2]
            # humanoid_obs = torch.cat([root_height.unsqueeze(-1), head_height.unsqueeze(-1)], dim=-1)
            root_coords = global_translations[env_ids, 0, :]
            head_coords = global_translations[env_ids, self.head_id, :]
            humanoid_obs = torch.cat([root_coords, head_coords], dim=-1)

        obs = compute_heading_observations(root_states, tar_dir, tar_speed, self.w_last)
        if self.use_current_pose_obs:
            obs = torch.cat([obs, humanoid_obs], dim=-1)
        self.direction_obs[env_ids] = obs
        self.inversion_obs = self.direction_obs

    def compute_reward(self, actions):
        root_pos = self.get_humanoid_root_states()[..., :3]
        self.rew_buf[:], output_dict = compute_heading_reward(
            root_pos, self._prev_root_pos, self._tar_dir, self._tar_speed, self.dt
        )
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

        other_log_terms = {
            "total_rew": self.rew_buf,
        }

        for rew_name, rew in other_log_terms.items():
            self.log_dict[f"{rew_name}_mean"] = rew.mean()
            # self.log_dict[f"{rew_name}_std"] = rew.std()

        self.last_unscaled_rewards: Dict[str, Tensor] = self.log_dict
        self.last_other_rewards = other_log_terms


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_heading_observations(
        root_states: Tensor, tar_dir: Tensor, tar_speed: Tensor, w_last: bool
) -> Tensor:
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, w_last)
    local_tar_dir = local_tar_dir[..., 0:2]

    tar_speed = tar_speed.unsqueeze(-1)

    obs = torch.cat([local_tar_dir, tar_speed], dim=-1)
    return obs


@torch.jit.script
def compute_heading_reward(
        root_pos: Tensor,
        prev_root_pos: Tensor,
        tar_dir: Tensor,
        tar_speed: Tensor,
        dt: float,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """
    Compute the reward for the steering task.
    The reward is based on the error in the target direction and speed.
    The reward is given by:
    reward = exp(-vel_err_scale * (tar_vel_err^2 + tangent_err_w * tangent_vel_err^2))
    where:
    tar_vel_err = target speed - current speed in the target direction
    tangent_vel_err = current speed in the tangent direction

    Args:
    root_pos: The root position of the humanoid
    prev_root_pos: The previous root position of the humanoid
    tar_dir: The target direction
    tar_speed: The target speed
    dt: The time step
    """
    vel_err_scale = 0.25
    tangent_err_w = 0.1

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_err = tar_speed - tar_dir_speed
    tangent_vel_err = tangent_speed
    dir_reward = torch.exp(
        -vel_err_scale
        * (
                tar_vel_err * tar_vel_err
                + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    speed_mask = tar_dir_speed < -0.5
    dir_reward[speed_mask] = 0
    output_dict = {
        "tar_dir_speed": tar_dir_speed,
        "tangent_speed": tangent_speed,
        "tar_vel_err": tar_vel_err,
        "tangent_vel_err": tangent_vel_err,
        "dir_reward": dir_reward,
    }
    return dir_reward, output_dict
