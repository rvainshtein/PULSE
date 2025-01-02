import numpy as np
import torch
from easydict import EasyDict
from hydra.utils import instantiate
from isaac_utils import torch_utils
from isaacgym import gymapi
from phys_anim.utils.scene_lib import Terrain
from torch import Tensor

from phc.env.tasks import humanoid_amp_task

TAR_ACTOR_ID = 1


class PMBase(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.config = cfg.env

        self.device = device_type + ':' + str(device_id)

        super().__init__(cfg, sim_params, physics_engine, device_type, device_id, headless)

        humanoid_asset = self.humanoid_assets[0]
        self.body_names = self.gym.get_asset_rigid_body_names(humanoid_asset)
        self.dof_names = self.gym.get_asset_dof_names(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        self.dt = self.control_freq_inv * self.sim_params.dt

        if "smpl" in self.config.asset.assetFileName:
            self.head_body_id = self.head_id = self.get_body_id("Head")
        else:
            self.head_body_id = self.head_id = self.get_body_id("head")

        self.create_terrain()
        self.build_termination_heights()

    def build_termination_heights(self):
        head_term_height = self.config.head_termination_height
        termination_height = self.config.termination_height

        termination_heights = np.array([termination_height] * self.num_bodies)

        termination_heights[self.head_id] = max(
            head_term_height, termination_heights[self.head_id]
        )

        asset_file = self.config.asset.assetFileName
        if "amp_humanoid_sword_shield" in asset_file:
            left_arm_id = self.get_body_id("left_lower_arm")

            shield_term_height = self.config.shield_termination_height
            termination_heights[left_arm_id] = max(
                shield_term_height, termination_heights[left_arm_id]
            )

        self.termination_heights = torch_utils.to_torch(
            termination_heights, device=self.device
        )

    def create_terrain(self):
        self.terrain: Terrain = instantiate(
            self.config.terrain,
            scene_lib=None,
            num_envs=self.num_envs,
            device=self.device,
        )

        self.only_terrain_height_samples = (
                torch.tensor(self.terrain.heightsamples)
                .view(self.terrain.tot_rows, self.terrain.tot_cols)
                .to(self.device)
                * self.terrain.vertical_scale
        )
        self.height_samples = (
                torch.tensor(self.terrain.heightsamples)
                .view(self.terrain.tot_rows, self.terrain.tot_cols)
                .to(self.device)
                * self.terrain.vertical_scale
        )

        self.non_termination_contact_body_ids = self.build_body_ids_tensor(
            self.config.robot.non_termination_contact_bodies
        )

    def build_body_ids_tensor(self, body_names):
        body_ids = []

        for body_name in body_names:
            body_id = self.body_names.index(body_name)
            assert (
                    body_id != -1
            ), f"Body part {body_name} not found in {self.body_names}"
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def get_body_positions(self):
        return self._rigid_body_pos.clone()

    def get_body_id(self, body_name):
        return self.gym.find_actor_rigid_body_handle(
            self.envs[0], self.humanoid_handles[0], body_name
        )

    def get_ground_heights(self, root_states):
        """
        This provides the height of the ground beneath the character.
        Not to confuse with the height-map projection that a sensor would see.
        Use this function for alignment between mocap and new terrains.
        """
        self.only_terrain_height_samples = (
                torch.tensor(self.terrain.heightsamples)
                .view(self.terrain.tot_rows, self.terrain.tot_cols)
                .to(self.device)
                * self.terrain.vertical_scale
        )

        height_samples = self.only_terrain_height_samples
        horizontal_scale = self.terrain.horizontal_scale

        return get_heights(
            root_states=root_states,
            height_samples=height_samples,
            horizontal_scale=horizontal_scale,
        )

    ###############################################################
    # Getters
    ###############################################################
    def get_humanoid_root_states(self):
        return self._humanoid_root_states[..., :7].clone()

    def get_bodies_contact_buf(self):
        return self._contact_forces.clone()

    def get_object_contact_buf(self):
        return self._contact_forces.clone()

    def get_bodies_state(self):
        body_pos = self._rigid_body_pos.clone()
        body_rot = self._rigid_body_rot.clone()
        body_vel = self._rigid_body_vel.clone()
        body_ang_vel = self._rigid_body_ang_vel.clone()

        return_dict = EasyDict(
            {
                "body_pos": body_pos,
                "body_rot": body_rot,
                "body_vel": body_vel,
                "body_ang_vel": body_ang_vel,
            }
        )
        return return_dict

    def get_dof_forces(self):
        return self.dof_force_tensor


@torch.jit.script
def get_heights(
        root_states: Tensor,
        height_samples: Tensor,
        horizontal_scale: float,
):
    num_envs = root_states.shape[0]

    points = root_states[..., :2].clone().reshape(num_envs, 1, 2)
    points = (points / horizontal_scale).long()
    px = points[:, :, 0].view(-1)
    py = points[:, :, 1].view(-1)
    px = torch.clip(px, 0, height_samples.shape[0] - 2)
    py = torch.clip(py, 0, height_samples.shape[1] - 2)

    heights1 = height_samples[px, py]
    heights2 = height_samples[px + 1, py + 1]
    heights = torch.max(heights1, heights2)

    return heights.view(num_envs, -1)
