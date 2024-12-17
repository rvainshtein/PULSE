# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLib

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pm.envs.pm_inversion import (
        MaskedMimicTaskHumanoid,
    )
else:
    MaskedMimicTaskHumanoid = object

# TODO: heading, pacer path follower, location, reach
#  task defines all parameters, by default they should be fully masked out.
#  allow task to control joints in several forms, e.g., pelvis position in X frames or all frames, etc...
#  see what works best, submit that. Just like PACER/ASE tried multiple rewards/paths until they figured
#  what works best.

# TODO: make sure no reset due to end of motion file, just keep going until the end of the episode.


class BaseMaskedMimicTask(MaskedMimicTaskHumanoid):  # type: ignore[misc]
    def __init__(self, config, device, motion_lib: Optional[MotionLib] = None):
        super().__init__(config, device, motion_lib=motion_lib)
        self.setup_task()

    ###############################################################
    # Set up environment
    ###############################################################
    def setup_task(self):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        if len(env_ids) > 0:
            self.reset_task(env_ids)

    def reset_task(self, env_ids):
        # Make sure in user-control mode that the history isn't visible.
        self.valid_hist_buf.set_all(False)

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_humanoid_obs(self, env_ids=None):
        humanoid_obs = super().compute_humanoid_obs(env_ids)

        # After the humanoid obs is called, we compute the task obs.
        # A task obs will have its own unique tensor.
        # We do not append the task obs to the humanoid obs, rather we allow the user to define the network structure
        # and how the task obs is used in the network.
        self.compute_task_obs()

        return humanoid_obs

    def compute_task_obs(self, env_ids=None):
        pass

    def mask_everything(self):
        # By Default mask everything out. Individual tasks will override this.
        self.masked_mimic_target_poses_masks[:] = False
        self.masked_mimic_target_bodies_masks[:] = False
        self.target_pose_obs_mask[:] = False
        self.object_bounding_box_obs_mask[:] = False
        self.motion_text_embeddings_mask[:] = False

    def update_task(self, actions):
        pass

    ###############################################################
    # Helpers
    ###############################################################
    def draw_task(self):
        return
