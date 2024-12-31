# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import Tuple, Dict

import torch
# from isaac_utils import rotations
from poselib.poselib.core import rotation3d as rotations
from torch import Tensor

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils_pm, torch_utils

from isaacgym import gymapi
from isaacgym import gymtorch
import isaacgym.torch_utils as itu
import numpy as np
from scipy.spatial.transform import Rotation as sRot
from phc.utils.flags import flags

TAR_ACTOR_ID = 1


class HumanoidDirection(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.config = cfg.env

        self._tar_speed_min = self.config.steering_params.tar_speed_min
        self._tar_speed_max = self.config.steering_params.tar_speed_max

        self._heading_change_steps_min = self.config.steering_params.change_steps_min
        self._heading_change_steps_max = self.config.steering_params.change_steps_max
        self._random_heading_probability = self.config.steering_params.random_heading_probability
        self._standard_heading_change = self.config.steering_params.standard_heading_change
        self._standard_speed_change = self.config.steering_params.standard_speed_change
        self._stop_probability = self.config.steering_params.stop_probability
        self.use_current_pose_obs = self.config.steering_params.get("use_current_pose_obs", False)

        self.pose_obs_size = 6 if self.use_current_pose_obs else 0  # 2 for root and head height, 6 for root and head coords, self.get_obs_size() for full humanoid pose

        # self.obs_size = self.config.steering_params.obs_size + self.pose_obs_size
        self.obs_size = self.config.steering_params.obs_size + self.pose_obs_size

        device = device_type + ':' + str(device_id)

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        if "smpl" in self.config.asset.assetFileName:
            self.head_id = self.get_body_id("Head")

        if self.use_current_pose_obs:
            self.direction_obs = torch.zeros(
                (
                    self.config.num_envs,
                    self.config.steering_params.obs_size + self.get_obs_size(),
                ),
                device=device,
                dtype=torch.float,
            )
        else:
            self.head_id = self.get_body_id("head")

        self.direction_obs = torch.zeros(
            (self.config.num_envs, self.obs_size),
            # 6 for root and head coords
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

        if (not self.headless):
            self._build_marker_state_tensors()

        return

    def get_body_id(self, body_name):
        return self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], body_name
        )

    def get_task_obs_size(self):
        task_obs_size = 0
        if self._enable_task_obs:
            task_obs_size = self.obs_size
        return task_obs_size

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        return

    def post_physics_step(self):
        super().post_physics_step()

        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return

    def _update_marker(self):
        # Not yet copied
        humanoid_root_pos = self._humanoid_root_states[..., 0:3]
        self._marker_pos[..., 0:2] = humanoid_root_pos[..., 0:2]
        self._marker_pos[..., 0] += 0.5 + 0.2 * self._tar_speed
        self._marker_pos[..., 2] = 0.0

        self._marker_rot[:] = 0
        self._marker_rot[:, -1] = 1.0

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids),
                                                     len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "heading_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0
        asset_options.linear_damping = 0
        asset_options.max_angular_velocity = 0
        asset_options.density = 0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._marker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        if (not self.headless):
            self._build_marker(env_id, env_ptr)

        return

    def _build_marker(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0
        default_pose.p.z = 0.0

        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", self.num_envs + 10,
                                              1, 0)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs

        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[...,
                              TAR_ACTOR_ID, :]
        self._marker_pos = self._marker_states[..., :3]
        self._marker_rot = self._marker_states[..., 3:7]
        self._marker_actor_ids = self._humanoid_actor_ids + itu.to_torch(self._marker_handles, device=self.device,
                                                                     dtype=torch.int32)

        return

    def _update_task(self):
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

    def _reset_task(self, env_ids):
        if len(env_ids) > 0:
            self.reset_heading_task(env_ids)
        super()._reset_task(env_ids)

    def _compute_flip_task_obs(self, normal_task_obs, env_ids):
        B, D = normal_task_obs.shape
        flip_task_obs = normal_task_obs.clone()
        flip_task_obs[:, 1] = -flip_task_obs[:, 1]

        return flip_task_obs

    def get_humanoid_root_states(self):
        # return self._humanoid_root_states[..., :7].clone()
        return self._humanoid_root_states[..., :7]

    def get_body_positions(self):
        # return self._rigid_body_pos.clone()
        return self._rigid_body_pos

    def _compute_task_obs(self, env_ids=None):
        super()._compute_task_obs(env_ids)

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

        obs = compute_heading_observations(root_states, tar_dir, tar_speed)
        if self.use_current_pose_obs:
            obs = torch.cat([obs, humanoid_obs], dim=-1)
        self.direction_obs[env_ids] = obs
        self.inversion_obs = self.direction_obs
        return obs

    def _compute_reward(self, actions):
        root_pos = self.get_humanoid_root_states()[..., :3]
        self.rew_buf[:], output_dict = compute_heading_reward(
            root_pos, self._prev_root_pos, self._tar_dir, self._tar_speed, self.dt
        )

        self.reward_raw = self.rew_buf[:, None]
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

        # other_log_terms = {
        #     "total_rew": self.rew_buf,
        # }
        #
        # for rew_name, rew in other_log_terms.items():
        #     self.log_dict[f"{rew_name}_mean"] = rew.mean()
        #     # self.log_dict[f"{rew_name}_std"] = rew.std()
        #
        # self.last_unscaled_rewards: Dict[str, Tensor] = self.log_dict
        # self.last_other_rewards = other_log_terms

    def _draw_task(self):
        self._update_marker()
        return

    def _reset_ref_state_init(self, env_ids):
        super()._reset_ref_state_init(env_ids)
        self.power_acc[env_ids] = 0

    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(
            env_ids)

        # ZL Hack: Forcing to always be facing the x-direction.
        if not self._has_upright_start:
            heading_rot_inv = torch_utils.calc_heading_quat_inv(humanoid_amp.remove_base_rot(root_rot))
        else:
            heading_rot_inv = torch_utils.calc_heading_quat_inv(root_rot)

        heading_rot_inv_repeat = heading_rot_inv[:, None].repeat(1, len(self._body_names), 1)
        root_rot = itu.quat_mul(heading_rot_inv, root_rot).clone()
        rb_pos = itu.quat_apply(heading_rot_inv_repeat, rb_pos - root_pos[:, None, :]).clone() + root_pos[:, None, :]
        rb_rot = itu.quat_mul(heading_rot_inv_repeat, rb_rot).clone()
        root_ang_vel = itu.quat_apply(heading_rot_inv, root_ang_vel).clone()
        root_vel = itu.quat_apply(heading_rot_inv, root_vel).clone()
        body_vel = itu.quat_apply(heading_rot_inv_repeat, body_vel).clone()

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel

    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_speed')):
            self._output_motion_target_speed = []

        tar_speed = self._tar_speed[0].cpu().numpy()
        self._output_motion_target_speed.append(tar_speed)

        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_speed) > 1):
            output_data = np.array(self._output_motion_target_speed)
            np.save('output/record_tar_speed.npy', output_data)

            self._output_motion_target_speed = []

        return


class HumanoidDirectionZ(HumanoidDirection):
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

# @torch.jit.script
def compute_heading_observations(
        root_states: Tensor, tar_dir: Tensor, tar_speed: Tensor
) -> Tensor:
    root_rot = root_states[:, 3:7]

    tar_dir3d = torch.cat([tar_dir, torch.zeros_like(tar_dir[..., 0:1])], dim=-1)
    # heading_rot = torch_utils_pm.calc_heading_quat_inv(root_rot, w_last=True)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    # local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d, w_last=True)
    local_tar_dir = rotations.quat_rotate(heading_rot, tar_dir3d)
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
