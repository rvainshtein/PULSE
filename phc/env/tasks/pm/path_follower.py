# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from typing import Tuple, Dict

import torch
from phys_anim.envs.env_utils.path_generator import PathGenerator

from phc.env.tasks.pm.base import PMBase
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


class HumanoidPathFollower(PMBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.config = cfg.env
        self._fail_dist = 4.0
        self._fail_height_dist = 0.5

        self.obs_size = self.config.path_follower_params.path_obs_size

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        self._prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )
        self.path_obs = torch.zeros(
            self.config.num_envs,
            self.config.path_follower_params.path_obs_size,
            device=self.device,
            dtype=torch.float,
        )
        self.reset_path_ids = torch.arange(
            self.config.num_envs, dtype=torch.long, device=self.device
        )
        self.build_path_generator()

        if "smpl" in self.config.asset.assetFileName:
            self.head_body_id = self.get_body_id("Head")
        else:
            self.head_body_id = self.get_body_id("head")

        if not self.headless:
            self._build_marker_state_tensors()

    # def get_body_id(self, body_name):
    #     return self.gym.find_actor_rigid_body_handle(
    #         self.envs[0], self.humanoid_handles[0], body_name
    #     )

    def build_path_generator(self):
        episode_dur = self.config.max_episode_length * self.dt
        self.path_generator = PathGenerator(
            self.config.path_follower_params.path_generator,
            self.device,
            self.num_envs,
            episode_dur,
            self.config.path_follower_params.path_generator.height_conditioned,
        )

    def fetch_path_samples(self, env_ids=None):
        # 5 seconds with 0.5 second intervals, 10 samples.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        timestep_beg = self.progress_buf[env_ids] * self.dt
        timesteps = torch.arange(
            self.config.path_follower_params.num_traj_samples,
            device=self.device,
            dtype=torch.float,
        )
        timesteps = timesteps * self.config.path_follower_params.traj_sample_timestep
        traj_timesteps = timestep_beg.unsqueeze(-1) + timesteps

        env_ids_tiled = torch.broadcast_to(env_ids.unsqueeze(-1), traj_timesteps.shape)

        traj_samples_flat = self.path_generator.calc_pos(
            env_ids_tiled.flatten(), traj_timesteps.flatten()
        )
        traj_samples = torch.reshape(
            traj_samples_flat,
            shape=(
                env_ids.shape[0],
                self.config.path_follower_params.num_traj_samples,
                traj_samples_flat.shape[-1],
            ),
        )

        return traj_samples

    # def get_body_id(self, body_name):
    #     return self.gym.find_actor_rigid_body_handle(
    #         self.envs[0], self.humanoid_handles[0], body_name
    #     )

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

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        # TODO: change
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

        for i in range(self.config.path_follower_params.num_traj_samples):
            marker_handle = self.gym.create_actor(
                env_ptr,
                self._marker_asset,
                default_pose,
                "marker",
                self.num_envs + 10,
                0,
                0,
            )
            self.gym.set_rigid_body_color(
                env_ptr,
                marker_handle,
                0,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(0.8, 0.0, 0.0),
            )
            self._marker_handles[env_id].append(marker_handle)

    def _build_marker_state_tensors(self):
        num_actors = self.root_states.shape[0] // self.num_envs
        self._marker_states = self.root_states.view(
            self.num_envs, num_actors, self.root_states.shape[-1]
        )[..., 1: (1 + self.config.path_follower_params.num_traj_samples), :]
        self._marker_pos = self._marker_states[..., :3]

        self._marker_actor_ids = self.humanoid_actor_ids.unsqueeze(
            -1
        ) + torch_utils.to_torch(
            self._marker_handles, dtype=torch.int32, device=self.device
        )
        self._marker_actor_ids = self._marker_actor_ids.flatten()

    def _reset_task(self, env_ids):
        super()._reset_task(env_ids)
        self.reset_path_ids = env_ids

    # def get_body_positions(self):
    #     return self._rigid_body_pos.clone()

    def _compute_task_obs(self, env_ids=None):
        bodies_positions = self.get_body_positions()

        if env_ids is None:
            root_states = self._humanoid_root_states
            head_position = bodies_positions[:, self.head_body_id, :]
            ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        else:
            root_states = self._humanoid_root_states[env_ids]
            head_position = bodies_positions[env_ids, self.head_body_id, :]
            ground_below_head = torch.min(bodies_positions[env_ids], dim=1).values[
                ..., 2
            ]

        if self.reset_path_ids is not None and len(self.reset_path_ids) > 0:
            reset_head_position = bodies_positions[
                                  self.reset_path_ids, self.head_body_id, :
                                  ]
            flat_reset_head_position = reset_head_position.view(-1, 3)
            ground_below_reset_head = self.get_ground_heights(
                bodies_positions[:, self.head_body_id, :2]
            )[self.reset_path_ids]
            flat_reset_head_position[..., 2] -= ground_below_reset_head.view(-1)
            self.path_generator.reset(self.reset_path_ids, flat_reset_head_position)

            self.reset_path_ids = None

        traj_samples = self.fetch_path_samples(env_ids)

        flat_head_position = head_position.view(-1, 3)
        flat_head_position[..., 2] -= ground_below_head.view(-1)

        obs = compute_path_observations(
            root_states,
            flat_head_position,
            traj_samples,
            True,
            self.config.path_follower_params.height_conditioned,
        )

        self.path_obs[env_ids] = obs
        self.inversion_obs = self.path_obs
        return obs

    def _compute_reward(self, actions):
        bodies_positions = self.get_body_positions()
        head_position = bodies_positions[:, self.head_body_id, :]

        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        ground_below_head = torch.min(bodies_positions, dim=1).values[..., 2]
        head_position[..., 2] -= ground_below_head.view(-1)

        self.rew_buf[:] = compute_path_reward(
            head_position, tar_pos, self.config.path_follower_params.height_conditioned
        )

    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        tar_pos = self.path_generator.calc_pos(env_ids, time)

        bodies_positions = self.get_body_positions()
        bodies_contact_buf = self.get_bodies_contact_buf()

        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.non_termination_contact_body_ids,
            bodies_positions,
            tar_pos,
            self.config.max_episode_length,
            self._fail_dist,
            self._fail_height_dist,
            self.config.enable_height_termination,
            self.config.path_follower_params.enable_path_termination,
            self.config.path_follower_params.height_conditioned,
            self.termination_heights
            + self.get_ground_heights(bodies_positions[:, self.head_body_id, :2]),
            self.head_body_id,
        )

    def _update_marker(self):
        traj_samples = self.fetch_path_samples().clone()
        self._marker_pos[:] = traj_samples
        if not self.config.path_follower_params.path_generator.height_conditioned:
            self._marker_pos[..., 2] = 0.8  # CT hack

        ground_below_marker = self.get_ground_heights(
            traj_samples[..., :2].view(-1, 2)
        ).view(traj_samples.shape[:-1])

        self._marker_pos[..., 2] += ground_below_marker

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(self._marker_actor_ids),
            len(self._marker_actor_ids),
        )

    def draw_task(self):
        cols = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)

        # bodies_positions = self.get_body_positions()

        self._update_marker()

        for i, env_ptr in enumerate(self.envs):
            verts = self.path_generator.get_traj_verts(i).clone()
            if not self.config.path_follower_params.path_generator.height_conditioned:
                verts[..., 2] = self.humanoid_root_states[i, 2]  # ZL Hack
            else:
                verts[..., 2] += self.get_ground_heights(
                    self.humanoid_root_states[i, :2].view(1, 2)
                ).view(-1)
            lines = torch.cat([verts[:-1], verts[1:]], dim=-1).cpu().numpy()
            curr_cols = np.broadcast_to(cols, [lines.shape[0], cols.shape[-1]])
            self.gym.add_lines(self.viewer, env_ptr, lines.shape[0], lines, curr_cols)

    def _reset_ref_state_init(self, env_ids):
        # TODO: change to something else??
        super()._reset_ref_state_init(env_ids)
        self.power_acc[env_ids] = 0

    def _sample_ref_state(self, env_ids):
        # TODO: change to something else??
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
        # TODO: change to something else??
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


class HumanoidPathFollowerZ(HumanoidPathFollower):
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
def compute_path_reward(head_pos, tar_pos, height_conditioned):
    # type: (Tensor, Tensor, bool) -> Tensor
    pos_err_scale = 2.0
    height_err_scale = 10.0

    pos_diff = tar_pos[..., 0:2] - head_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    height_diff = tar_pos[..., 2] - head_pos[..., 2]
    height_err = height_diff * height_diff

    pos_reward = torch.exp(-pos_err_scale * pos_err)
    height_reward = torch.exp(-height_err_scale * height_err)

    if height_conditioned:
        reward = (pos_reward + height_reward) * 0.5
    else:
        reward = pos_reward

    return reward


@torch.jit.script
def compute_humanoid_reset(
        reset_buf,
        progress_buf,
        contact_buf,
        non_termination_contact_body_ids,
        rigid_body_pos,
        tar_pos,
        max_episode_length,
        fail_dist,
        fail_height_dist,
        enable_early_termination,
        enable_path_termination,
        enable_height_termination,
        termination_heights,
        head_body_id,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, bool, bool, bool, Tensor, int) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, non_termination_contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, non_termination_contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
    else:
        has_fallen = progress_buf < -1

    if enable_path_termination:
        head_pos = rigid_body_pos[..., head_body_id, :]
        tar_delta = tar_pos - head_pos
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_overall_fail = tar_dist_sq > fail_dist * fail_dist

        if enable_height_termination:
            tar_height = tar_pos[..., 2]
            height_delta = tar_height - head_pos[..., 2]
            tar_head_dist_sq = height_delta * height_delta
            tar_height_fail = tar_head_dist_sq > fail_height_dist * fail_height_dist
            tar_height_fail *= progress_buf > 20

            tar_fail = torch.logical_or(tar_overall_fail, tar_height_fail)
        else:
            tar_fail = tar_overall_fail
    else:
        tar_fail = progress_buf < -1

    has_failed = torch.logical_or(has_fallen, tar_fail)

    terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated


@torch.jit.script
def compute_path_observations(
        root_states: Tensor,
        head_states: Tensor,
        traj_samples: Tensor,
        w_last: bool,
        height_conditioned: bool,
) -> Tensor:
    root_rot = root_states[:, 3:7]
    # heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    heading_rot_exp = torch.broadcast_to(
        heading_rot.unsqueeze(-2),
        (heading_rot.shape[0], traj_samples.shape[1], heading_rot.shape[1]),
    )
    heading_rot_exp = torch.reshape(
        heading_rot_exp,
        (heading_rot_exp.shape[0] * heading_rot_exp.shape[1], heading_rot_exp.shape[2]),
    )

    traj_samples_delta = traj_samples - head_states.unsqueeze(-2)

    traj_samples_delta_flat = torch.reshape(
        traj_samples_delta,
        (
            traj_samples_delta.shape[0] * traj_samples_delta.shape[1],
            traj_samples_delta.shape[2],
        ),
    )

    # local_traj_pos = rotations.quat_rotate(heading_rot_exp, traj_samples_delta_flat, w_last)
    local_traj_pos = rotations.quat_rotate(heading_rot_exp, traj_samples_delta_flat)
    if not height_conditioned:
        local_traj_pos = local_traj_pos[..., 0:2]

    obs = torch.reshape(
        local_traj_pos,
        (traj_samples.shape[0], traj_samples.shape[1] * local_traj_pos.shape[1]),
    )
    return obs
