# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

import env.tasks.humanoid as humanoid
import env.tasks.humanoid_amp as humanoid_amp
import env.tasks.humanoid_amp_task as humanoid_amp_task
from utils import torch_utils
from isaac_utils import rotations

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym import gymtorch
from isaacgym.torch_utils import *

class HumanoidReach(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self._tar_change_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._tar_reach_steps = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.int64
        )
        self._tar_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )

        self._last_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.bool
        )
        self.w_last = True
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)

        reach_body_name = cfg.env.reach_body_name
        self._reach_body_id = self._build_reach_body_id_tensor(self.envs[0], self.humanoid_handles[0], reach_body_name)
        
        if (not self.headless):
            self._build_marker_state_tensors()

        return
    
    def _sample_ref_state(self, env_ids):
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel = super()._sample_ref_state(env_ids)
        root_pos[..., :2] = 0.0 # Set the root position to be zero
        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel,  rb_pos, rb_rot, body_vel, body_ang_vel

    def get_task_obs_size(self):
        obs_size = 0
        if (self._enable_task_obs):
            obs_size = 3
        return obs_size

    def post_physics_step(self):
        super().post_physics_step()

        if (humanoid_amp.HACK_OUTPUT_MOTION):
            self._hack_output_motion_target()

        return

    def _update_marker(self):
        self._marker_pos[..., :] = self._tar_pos
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(self._marker_actor_ids), len(self._marker_actor_ids))
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_marker_asset(self):
        asset_root = "phc/data/assets/urdf/"
        asset_file = "location_marker.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
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
        
        marker_handle = self.gym.create_actor(env_ptr, self._marker_asset, default_pose, "marker", env_id, 2, 2)
        self.gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
        self._marker_handles.append(marker_handle)

        return

    def _build_marker_state_tensors(self):
        num_actors = self._root_states.shape[0] // self.num_envs
        self._marker_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        self._marker_pos = self._marker_states[..., :3]
        
        self._marker_actor_ids = self._humanoid_actor_ids + 1

        return
    
    def _build_reach_body_id_tensor(self, env_ptr, actor_handle, body_name):
        body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
        assert(body_id != -1)
        body_id = to_torch(body_id, device=self.device, dtype=torch.long)
        return body_id

    def _update_task(self):
        reset_task_mask = self.progress_buf >= self._tar_change_steps
        rest_env_ids = reset_task_mask.nonzero(as_tuple=False).flatten()
        if len(rest_env_ids) > 0:
            self._reset_task(rest_env_ids)
        return

    def _reset_task(self, env_ids):
        if len(env_ids) > 0:
            # Make sure the test has started + agent started from a valid position (if it failed, then it's not valid)
            active_envs = (self._current_accumulated_errors[env_ids] > 0) & (
                ~self._last_failures[env_ids]
            )
            average_distances = self._current_accumulated_errors[env_ids][
                                    active_envs
                                ] / (
                                        self._last_length[env_ids][active_envs]
                                        - self._tar_reach_steps[env_ids][active_envs]
                                )
            self._distances.extend(average_distances.cpu().tolist())
            self._current_accumulated_errors[env_ids] = 0
            self._failures.extend(
                (self._current_failures[env_ids][active_envs] > 0).cpu().tolist()
            )
            self._last_failures[env_ids] = self._current_failures[env_ids] > 0
            self._current_failures[env_ids] = 0

        super().reset_task(env_ids)
        n = len(env_ids)

        rand_pos = torch.rand([n, 3], device=self.device)
        rand_pos[..., 0:2] = self.config.reach_params.tar_dist_max * (
                1.5 * rand_pos[..., 0:2] - 0.75
        )
        rand_pos[..., 2] = (
                                   self.config.reach_params.tar_height_max
                                   - self.config.reach_params.tar_height_min
                           ) * rand_pos[..., 2] + self.config.reach_params.tar_height_min

        change_steps = torch.randint(
            low=self.config.reach_params.change_steps_min,
            high=self.config.reach_params.change_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )
        reach_steps = torch.randint(
            low=self.config.reach_params.reach_steps_min,
            high=self.config.reach_params.reach_steps_max,
            size=(n,),
            device=self.device,
            dtype=torch.int64,
        )
        # min with change_steps to avoid reaching AFTER changing
        reach_steps = torch.min(reach_steps, change_steps)

        bodies_positions = self.get_body_positions()
        root_pos = bodies_positions[env_ids, 0, :]
        root_pos[:, -1:] = 0

        marker_pos = root_pos + rand_pos
        marker_pos[:, -1:] += self.get_ground_heights(marker_pos[:, :2]).view(-1, 1)

        # Marker position is represented relative to the character pos, without terrains.
        self._tar_pos[env_ids, :] = marker_pos
        self._tar_change_steps[env_ids] = self.progress_buf[env_ids] + change_steps
        self._tar_reach_steps[env_ids] = self.progress_buf[env_ids] + reach_steps

    def _compute_task_obs(self, env_ids=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_pos = self._tar_pos
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_pos = self._tar_pos[env_ids]
        
        reach_obs = compute_location_observations(root_states, tar_pos, self.w_last)
        return obs

    def _compute_reward(self, actions):
        reach_body_pos = self._rigid_body_pos[:, self._reach_body_id, :]
        root_rot = self._humanoid_root_states[..., 3:7]
        
        self.rew_buf[:] = compute_reach_reward(reach_body_pos, root_rot,
                                                 self._tar_pos, self._tar_speed,
                                                 self.dt)
        return

    def compute_failures_and_distances(self):
        body_pos = self.get_bodies_state().body_pos
        reach_actual_pos = body_pos[:, self.reach_body_id, :]
        goal_pos = self._tar_pos
        distance_to_target = torch.norm(reach_actual_pos - goal_pos, dim=-1).view(
            self.num_envs
        )
        measurement_started = self._tar_reach_steps < self.progress_buf
        measurement_not_started = ~measurement_started
        self._current_accumulated_errors[measurement_started] += distance_to_target[
            measurement_started
        ]
        self._current_failures[measurement_started] += (
                distance_to_target[measurement_started] > 0.5
        )
        self._current_failures[measurement_not_started] = 0
        self._current_accumulated_errors[measurement_not_started] = 0
        self._last_length[:] = self.progress_buf[:]

    def _draw_task(self):
        self._update_marker()
        
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = self._rigid_body_pos[:, self._reach_body_id, :]
        ends = self._tar_pos
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return

    def _hack_output_motion_target(self):
        if (not hasattr(self, '_output_motion_target_pos')):
            self._output_motion_target_pos = []

        tar_pos = self._tar_pos[0].cpu().numpy()
        self._output_motion_target_pos.append(tar_pos)

        reset = self.reset_buf[0].cpu().numpy() == 1

        if (reset and len(self._output_motion_target_pos) > 1):
            output_data = np.array(self._output_motion_target_pos)
            np.save('output/record_tar_motion.npy', output_data)

            self._output_motion_target_pos = []

        return
    
class HumanoidReachZ(HumanoidReach):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        super().__init__(cfg=cfg, sim_params=sim_params, physics_engine=physics_engine, device_type=device_type, device_id=device_id, headless=headless)
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
def compute_location_observations(root_states, tar_pos, w_last=True):
    # type: (Tensor, Tensor, bool) -> Tensor
    root_rot = root_states[:, 3:7]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)
    local_tar_pos = rotations.quat_rotate(heading_rot, tar_pos, w_last)

    obs = local_tar_pos
    return obs

@torch.jit.script
def compute_reach_reward(reach_body_pos, root_rot, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    pos_err_scale = 4.0
    
    pos_diff = tar_pos - reach_body_pos
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)
    
    reward = pos_reward

    return reward