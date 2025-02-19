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
from typing import Optional, Dict

from isaac_utils import torch_utils
from isaac_utils.rotations import quat_mul
from isaacgym import gymapi, gymtorch
from isaac_utils.torch_utils import *
from torch import Tensor

from phc.env.tasks.pm.base import PMBase


class MaskedMimicStrike(PMBase):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        if not self.headless:
            self._build_marker_state_tensors()

        self._tar_dist_min = self.config.strike_params.tar_dist_min
        self._tar_dist_max = self.config.strike_params.tar_dist_max
        self._near_dist = self.config.strike_params.near_dist
        self._near_prob = self.config.strike_params.near_prob
        self._tar_speed = self.config.strike_params.get("tar_speed", 1.0)
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)

        strike_body_names = self.config.strike_params.strike_body_names
        self._strike_body_ids = self.build_body_ids_tensor(strike_body_names)
        self._build_target_tensors()
        self._current_successes = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 15
        return obs_size

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._target_handles = []
        self._marker_handles = [[] for _ in range(num_envs)]

        self._load_target_asset()

        super()._create_envs(num_envs, spacing, num_per_row)

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self._build_target(env_id, env_ptr)

    def _load_target_asset(self):
        asset_root = "phys_anim/data/assets/urdf/"
        asset_file = "strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._target_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        target_handle = self.gym.create_actor(env_ptr, self._target_asset, default_pose, "target", col_group,
                                              col_filter, segmentation_id)
        self._target_handles.append(target_handle)

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]

        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1

        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]

        return

    def _reset_actors(self, env_ids):
        super()._reset_actors(env_ids)
        self._reset_strike_target(env_ids)

    def _reset_task(self, env_ids=None):
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = self._last_length[env_ids] > 0
            average_distances = (self._current_accumulated_errors[env_ids][active_envs] /
                                 self._last_length[env_ids][active_envs])
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_successes[env_ids][active_envs] == 0).cpu().tolist()
            )
            self._current_successes[env_ids] = 0
            self._reset_strike_target(env_ids)
        super()._reset_task(env_ids)

    def _reset_strike_target(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        root_states = self.get_humanoid_root_states()

        n = len(env_ids)

        init_near = torch.rand([n], dtype=self._target_states.dtype,
                               device=self._target_states.device) < self._near_prob
        dist_max = self._tar_dist_max * torch.ones([n], dtype=self._target_states.dtype,
                                                   device=self._target_states.device)
        dist_max[init_near] = self._near_dist
        rand_dist = (dist_max - self._tar_dist_min) * torch.rand([n], dtype=self._target_states.dtype,
                                                                 device=self._target_states.device) + self._tar_dist_min

        rand_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        self._target_states[env_ids, 0] = rand_dist * torch.cos(rand_theta) + root_states[env_ids, 0]
        self._target_states[env_ids, 1] = rand_dist * torch.sin(rand_theta) + root_states[env_ids, 1]
        self._target_states[env_ids, 2] = 0.9

        rand_rot_theta = 2 * np.pi * torch.rand([n], dtype=self._target_states.dtype, device=self._target_states.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=self._target_states.dtype, device=self._target_states.device)
        rand_rot = quat_from_angle_axis(rand_rot_theta, axis, w_last=self.w_last)

        self._target_states[env_ids, 3:7] = rand_rot
        self._target_states[env_ids, 7:10] = 0.0
        self._target_states[env_ids, 10:13] = 0.0

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)

        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32),
                                                     len(env_ids_int32))

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self._prev_root_pos[:] = self.get_humanoid_root_states()[..., 0:3]

    def _compute_task_obs(self, env_ids=None):
        super()._compute_task_obs(env_ids)
        if (env_ids is None):
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        root_states = self.get_humanoid_root_states()[env_ids]
        tar_states = self._target_states[env_ids]

        obs = compute_strike_observations(root_states, tar_states)

    def compute_reward(self, actions):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        char_root_state = self._humanoid_root_states
        strike_body_vel = self._rigid_body_vel[..., self._strike_body_ids[0], :]

        self.rew_buf[:], output_dict = compute_strike_reward(tar_pos,
                                                             tar_rot,
                                                             char_root_state,
                                                             self._prev_root_pos,
                                                             self.dt,
                                                             self._tar_speed,
                                                             self.w_last)

        # self.log_dict.update(output_dict)
        # # need these at the end of every compute_reward function
        self.compute_failures_and_distances()
        self.accumulate_errors()

    def compute_failures_and_distances(self):
        tar_pos = self._target_states[..., 0:3]
        tar_rot = self._target_states[..., 3:7]
        up = torch.zeros_like(tar_pos)
        up[..., -1] = 1
        tar_up = quat_rotate(tar_rot, up, w_last=self.w_last)
        tar_rot_err = torch.sum(up * tar_up, dim=-1)

        distance_to_target = torch.norm(self._humanoid_root_states[..., 0:3] - tar_pos, dim=-1)

        self._current_accumulated_errors[:] += distance_to_target
        self._current_successes[:] = tar_rot_err < 0.2
        self._last_length[:] = self.progress_buf[:]

    def compute_reset(self):
        bodies_positions = self.get_body_positions()

        bodies_positions[..., 2] -= (
            torch.min(bodies_positions, dim=1).values[:, 2].view(-1, 1)
        )

        termination_heights = self.termination_heights + self.get_ground_heights(
            bodies_positions[:, self.head_body_id, :2])
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces,
                                                                           self.non_termination_contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self._tar_contact_forces,
                                                                           self._strike_body_ids,
                                                                           self.config.max_episode_length,
                                                                           self.config.enable_height_termination,
                                                                           termination_heights, )

    def _draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._humanoid_root_states[..., 0:3]
        ends = self._target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_strike_observations(root_states, tar_states, w_last=True):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last=w_last)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = quat_rotate(heading_rot, local_tar_pos, w_last=w_last)
    local_tar_vel = quat_rotate(heading_rot, tar_vel, w_last=w_last)
    local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel, w_last=w_last)

    local_tar_rot = quat_mul(heading_rot, tar_rot, w_last=w_last)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot, w_last=w_last)

    obs = torch.cat([local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1)
    return obs


@torch.jit.script
def compute_strike_reward(tar_pos, tar_rot, root_state, prev_root_pos, dt, tar_speed, w_last=True):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, bool) -> Tuple[Tensor, Dict[str, Tensor]]
    # tar_speed = 1.0
    # tar_speed = 2.0
    vel_err_scale = 4.0

    tar_rot_w = 0.6
    vel_reward_w = 0.4

    up = torch.zeros_like(tar_pos)
    up[..., -1] = 1
    tar_up = quat_rotate(tar_rot, up, w_last=w_last)
    tar_rot_err = torch.sum(up * tar_up, dim=-1)
    tar_rot_r = torch.clamp_min(1.0 - tar_rot_err, 0.0)

    root_pos = root_state[..., 0:3]
    tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err_rel = tar_vel_err / tar_speed  # it's 1.0 always
    tar_vel_err_rel = torch.clamp_min(tar_vel_err_rel, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err_rel * tar_vel_err_rel))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0

    reward = tar_rot_w * tar_rot_r + vel_reward_w * vel_reward

    succ = tar_rot_err < 0.2
    reward = torch.where(succ, torch.ones_like(reward), reward)

    output_dict = {
        "tar_rot_err": tar_rot_err,
        "tar_rot_r": tar_rot_r,
        "vel_reward": vel_reward,
    }
    return reward, output_dict


@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, non_termination_contact_body_ids, rigid_body_pos,
                           tar_contact_forces,
                           strike_body_ids, max_episode_length, enable_early_termination, termination_heights):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, Tensor) -> Tuple[Tensor, Tensor]
    contact_force_threshold = 1.0

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

        tar_has_contact = torch.any(torch.abs(tar_contact_forces[..., 0:2]) > contact_force_threshold, dim=-1)
        # strike_body_force = contact_buf[:, strike_body_id, :]
        # strike_body_has_contact = torch.any(torch.abs(strike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_force = masked_contact_buf
        nonstrike_body_force[:, strike_body_ids, :] = 0
        nonstrike_body_has_contact = torch.any(torch.abs(nonstrike_body_force) > contact_force_threshold, dim=-1)
        nonstrike_body_has_contact = torch.any(nonstrike_body_has_contact, dim=-1)

        tar_fail = torch.logical_and(tar_has_contact, nonstrike_body_has_contact)

        has_failed = torch.logical_or(has_fallen, tar_fail)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_failed *= (progress_buf > 1)
        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated
