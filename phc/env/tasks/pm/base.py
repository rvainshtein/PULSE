import numpy as np
import torch
from easydict import EasyDict
from hydra.utils import instantiate
from isaac_utils import torch_utils
from isaacgym import gymapi
from phys_anim.utils.scene_lib import Terrain
from torch import Tensor
from isaacgym.torch_utils import *

from phc.env.tasks import humanoid_amp_task
from phc.utils.flags import flags
TAR_ACTOR_ID = 1
from isaacgym.terrain_utils import *
from phc.utils.draw_utils import *
from tqdm import tqdm
class Terrain:
    def __init__(self, cfg, num_robots, device) -> None:

        self.type = cfg["terrainType"]
        self.device = device
        if self.type in ["none", 'plane']:
            return
        self.horizontal_scale = 0.1 # resolution 0.1
        self.vertical_scale = 0.005
        self.border_size = 40
        self.env_length = cfg["mapLength"]
        self.env_width = cfg["mapWidth"]
        self.proportions = [
            np.sum(cfg["terrainProportions"][:i + 1])
            for i in range(len(cfg["terrainProportions"]))
        ]

        self.env_rows = cfg["numLevels"]
        self.env_cols = cfg["numTerrains"]
        self.num_maps = self.env_rows * self.env_cols
        self.env_origins = np.zeros((self.env_rows, self.env_cols, 3))

        self.width_per_env_pixels = int(self.env_width / self.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length /
                                         self.horizontal_scale)

        self.border = int(self.border_size / self.horizontal_scale)
        self.tot_cols = int(
            self.env_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(
            self.env_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.walkable_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        if cfg["curriculum"]:
            self.curiculum(num_robots,
                           num_terrains=self.env_cols,
                           num_levels=self.env_rows)
        else:
            self.randomized_terrain()
        self.heightsamples = torch.from_numpy(self.height_field_raw).to(self.device) # ZL: raw height field, first dimension is x, second is y
        self.walkable_field = torch.from_numpy(self.walkable_field_raw).to(self.device)
        self.vertices, self.triangles = convert_heightfield_to_trimesh(self.height_field_raw, self.horizontal_scale, self.vertical_scale,cfg["slopeTreshold"])
        self.sample_extent_x = int((self.tot_rows - self.border * 2) * self.horizontal_scale)
        self.sample_extent_y = int((self.tot_cols - self.border * 2) * self.horizontal_scale)

        coord_x, coord_y = torch.where(self.walkable_field == 0)
        coord_x_scale = coord_x * self.horizontal_scale
        coord_y_scale = coord_y * self.horizontal_scale
        walkable_subset = torch.logical_and(
                torch.logical_and(coord_y_scale < coord_y_scale.max() - self.border * self.horizontal_scale, coord_x_scale < coord_x_scale.max() - self.border * self.horizontal_scale),
                torch.logical_and(coord_y_scale > coord_y_scale.min() + self.border * self.horizontal_scale, coord_x_scale > coord_x_scale.min() +  self.border * self.horizontal_scale)
            )
        # import ipdb; ipdb.set_trace()
        # joblib.dump(self.walkable_field_raw, "walkable_field.pkl")

        self.coord_x_scale = coord_x_scale[walkable_subset]
        self.coord_y_scale = coord_y_scale[walkable_subset]
        self.num_samples = self.coord_x_scale.shape[0]


    def sample_valid_locations(self, max_num_envs, env_ids, group_num_people = 16, sample_groups = False):
        if sample_groups:
            num_groups = max_num_envs// group_num_people
            group_centers = torch.stack([torch_rand_float(0., self.sample_extent_x, (num_groups, 1),device=self.device).squeeze(1), torch_rand_float(0., self.sample_extent_y, (num_groups, 1),device=self.device).squeeze(1)], dim = -1)
            group_diffs = torch.stack([torch_rand_float(-8., 8, (num_groups, group_num_people) ,device=self.device), torch_rand_float(8., -8, (num_groups, group_num_people),device=self.device)], dim = -1)
            valid_locs = (group_centers[:, None, ] + group_diffs).reshape(max_num_envs, -1)

            if not env_ids is None:
                valid_locs = valid_locs[env_ids]
        else:
            num_envs = env_ids.shape[0]
            idxes = np.random.randint(0, self.num_samples, size=num_envs)
            valid_locs = torch.stack([self.coord_x_scale[idxes], self.coord_y_scale[idxes]], dim = -1)

        return valid_locs

    def world_points_to_map(self, points):
        points = (points / self.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.heightsamples.shape[0] - 2)
        py = torch.clip(py, 0, self.heightsamples.shape[1] - 2)
        return px, py


    def sample_height_points(self, points, root_states = None, root_points=None, env_ids = None, num_group_people = 512, group_ids = None):
        B, N, C = points.shape
        px, py = self.world_points_to_map(points)
        heightsamples = self.heightsamples.clone()
        if env_ids is None:
            env_ids = torch.arange(B).to(points).long()

        if not root_points is None:
            # Adding human root points to the height field
            max_num_envs, num_root_points, _ = root_points.shape
            root_px, root_py = self.world_points_to_map(root_points)
            num_groups = int(root_points.shape[0]/num_group_people)
            heightsamples_group = heightsamples[None, ].repeat(num_groups, 1, 1)

            root_px, root_py = root_px.view(-1, num_group_people * num_root_points), root_py.view(-1, num_group_people *  num_root_points)
            px, py = px.view(-1, N), py.view(-1, N)
            heights = torch.zeros_like(px)

            if not root_states is None:
                linear_vel = root_states[:, 7:10] # This contains ALL the linear velocities
                root_rot = root_states[:, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                velocity_map = torch.zeros([px.shape[0], px.shape[1], 2]).to(root_states)
                velocity_map_group = torch.zeros(heightsamples_group.shape + (3,)).to(points)

            for idx in range(num_groups):
                heightsamples_group[idx][root_px[idx], root_py[idx]] += torch.tensor(1.7 / self.vertical_scale).short()
                group_mask_env_ids = group_ids[env_ids] == idx # agents to select for this group from the current env_ids
                # if sum(group_mask) == 0:
                #     continue
                group_px, group_py = px[group_mask_env_ids].view(-1), py[group_mask_env_ids].view(-1)
                heights1 = heightsamples_group[idx][group_px, group_py]
                heights2 = heightsamples_group[idx][group_px + 1, group_py + 1]
                heights_group = torch.min(heights1, heights2)
                heights[group_mask_env_ids] = heights_group.view(-1, N).long()

                if not root_states is None:
                    # First update the map with the velocity
                    group_mask_all = group_ids == idx
                    env_ids_in_group = env_ids[group_mask_env_ids]
                    group_linear_vel = linear_vel[group_mask_all]
                    velocity_map_group[idx, root_px[idx], root_py[idx], :] = group_linear_vel.repeat(1, root_points.shape[1]).view(-1, 3)

                    # Sample the points for each agent's px and py
                    vel_group = velocity_map_group[idx][group_px, group_py]
                    vel_group = vel_group.view(-1, N, 3)
                    vel_group -= linear_vel[env_ids_in_group, None]  # for each agent's velocity map, minus it's own velocity to get the relative velocity
                    group_heading_rot = heading_rot[env_ids_in_group]

                    group_vel_idv = torch_utils.my_quat_rotate(
                        group_heading_rot.repeat(1, N).view(-1, 4),
                        vel_group.view(-1, 3)
                    )  # Global velocity transform. for ALL of the elements in the group.
                    group_vel_idv = group_vel_idv.view(-1, N, 3)[..., :2]
                    velocity_map[group_mask_env_ids] = group_vel_idv
            if root_states is None:
                return heights * self.vertical_scale
            else:
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)

        else:
            heights1 = heightsamples[px, py]
            heights2 = heightsamples[px + 1, py + 1]
            heights = torch.min(heights1, heights2)

            if root_states is None:
                return heights * self.vertical_scale
            else:
                velocity_map = torch.zeros((B, N, 2)).to(points)
                linear_vel = root_states[env_ids, 7:10]
                root_rot = root_states[env_ids, 3:7]
                heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
                linear_vel_ego = torch_utils.my_quat_rotate(heading_rot, linear_vel)
                velocity_map[:] = velocity_map[:] - linear_vel_ego[:, None, :2] # Flip velocity to be in agent's point of view
                heights = (heights * self.vertical_scale).view(B, -1, 1)
                return torch.cat([heights, velocity_map], dim = -1)

    # def sample_height_points(self, points, root_points=None):
    #     # Ugly but correct solution
    #     B, N, _ = points.shape
    #     px, py = self.world_points_to_map(points)

    #     if not root_points is None:
    #         # Adding human root points to the height field
    #         root_px, root_py = self.world_points_to_map(root_points)
    #         root_px, root_py = root_px.view(B, -1), root_py.view(B, -1)
    #         heights_acc = []
    #         for curr_agent in range(B):
    #             px, py = px.view(B, -1), py.view(B, -1)
    #             heightsamples = self.heightsamples.clone()
    #             mask = torch.ones(B).bool()
    #             mask[curr_agent] = False
    #             heightsamples[root_px[mask].flatten(), root_py[mask].flatten()] += torch.tensor(1.7 / self.vertical_scale).short()
    #             heights1 = heightsamples[px[curr_agent], py[curr_agent]]
    #             heights2 = heightsamples[px[curr_agent] + 1, py[curr_agent] + 1]
    #             heights = torch.min(heights1, heights2)
    #             heights_acc.append(heights)
    #         heights = torch.stack(heights_acc, dim=0)

    #     else:
    #         heightsamples = self.heightsamples.clone()
    #         heights1 = heightsamples[px, py]
    #         heights2 = heightsamples[px + 1, py + 1]
    #         heights = torch.min(heights1, heights2)
    #     return heights

    def randomized_terrain(self):
        for k in range(self.num_maps):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.env_rows, self.env_cols))

            # Heightfield coordinate system from now on
            start_x = self.border + i * self.length_per_env_pixels
            end_x = self.border + (i + 1) * self.length_per_env_pixels
            start_y = self.border + j * self.width_per_env_pixels
            end_y = self.border + (j + 1) * self.width_per_env_pixels

            terrain = SubTerrain("terrain",
                                 width=self.width_per_env_pixels,
                                 length=self.width_per_env_pixels,
                                 vertical_scale=self.vertical_scale,
                                 horizontal_scale=self.horizontal_scale)
            choice = np.random.uniform(0, 1)
            difficulty = np.random.uniform(0.1, 1)
            slope = difficulty * 0.7
            discrete_obstacles_height = 0.025 + difficulty * 0.15
            stepping_stones_size = 2 - 1.8 * difficulty
            step_height = 0.05 + 0.175 * difficulty
            if choice < self.proportions[0]:
                if choice < 0.05:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            elif choice < self.proportions[1]:
                if choice < 0.15:
                    slope *= -1
                pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
                random_uniform_terrain(terrain,
                                       min_height=-0.1,
                                       max_height=0.1,
                                       step=0.025,
                                       downsampled_scale=0.2)
            elif choice < self.proportions[3]:
                if choice < self.proportions[2]:
                    step_height *= -1
                pyramid_stairs_terrain(terrain,
                                       step_width=0.31,
                                       step_height=step_height,
                                       platform_size=3.)
            elif choice < self.proportions[4]:
                discrete_obstacles_terrain(terrain,
                                           discrete_obstacles_height,
                                           1.,
                                           2.,
                                           40,
                                           platform_size=3.)
            elif choice < self.proportions[5]:
                stepping_stones_terrain(terrain,
                                        stone_size=stepping_stones_size,
                                        stone_distance=0.1,
                                        max_height=0.,
                                        platform_size=3.)
            elif choice < self.proportions[6]:
                poles_terrain(terrain=terrain, difficulty=difficulty)
                self.walkable_field_raw[start_x:end_x, start_y:end_y] = (terrain.height_field_raw != 0)

            elif choice < self.proportions[7]:
                # plain walking terrain
                pass

            self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

            env_origin_x = (i + 0.5) * self.env_length
            env_origin_y = (j + 0.5) * self.env_width
            x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
            x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
            y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
            y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
            env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2]) * self.vertical_scale
            self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]
        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)

    def curiculum(self, num_robots, num_terrains, num_levels):
        num_robots_per_map = int(num_robots / num_terrains)
        left_over = num_robots % num_terrains
        idx = 0
        for j in tqdm(range(num_terrains)):
            for i in range(num_levels):
                terrain = SubTerrain("terrain",
                                     width=self.width_per_env_pixels,
                                     length=self.width_per_env_pixels,
                                     vertical_scale=self.vertical_scale,
                                     horizontal_scale=self.horizontal_scale)
                difficulty = i / num_levels
                choice = j / num_terrains

                slope = difficulty * 0.7
                step_height = 0.05 + 0.175 * difficulty
                discrete_obstacles_height = 0.025 + difficulty * 0.15
                stepping_stones_size = 2 - 1.8 * difficulty

                start_x = self.border + i * self.length_per_env_pixels
                end_x = self.border + (i + 1) * self.length_per_env_pixels
                start_y = self.border + j * self.width_per_env_pixels
                end_y = self.border + (j + 1) * self.width_per_env_pixels

                if choice < self.proportions[0]:
                    if choice < 0.05:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                elif choice < self.proportions[1]:
                    if choice < 0.15:
                        slope *= -1
                    pyramid_sloped_terrain(terrain,
                                           slope=slope,
                                           platform_size=3.)
                    random_uniform_terrain(terrain,
                                           min_height=-0.1,
                                           max_height=0.1,
                                           step=0.025,
                                           downsampled_scale=0.2)
                elif choice < self.proportions[3]:
                    if choice < self.proportions[2]:
                        step_height *= -1
                    pyramid_stairs_terrain(terrain,
                                           step_width=0.31,
                                           step_height=step_height,
                                           platform_size=3.)
                elif choice < self.proportions[4]:
                    discrete_obstacles_terrain(terrain,
                                               discrete_obstacles_height,
                                               1.,
                                               2.,
                                               40,
                                               platform_size=3.)
                elif choice < self.proportions[5]:
                    stepping_stones_terrain(terrain,
                                            stone_size=stepping_stones_size,
                                            stone_distance=0.1,
                                            max_height=0.,
                                            platform_size=3.)
                elif choice < self.proportions[6]:
                    poles_terrain(terrain=terrain, difficulty=difficulty)
                    self.walkable_field_raw[start_x:end_x, start_y:end_y] = (terrain.height_field_raw != 0)
                    # self.walkable_field_raw[start_x:end_x, start_y:end_y] = 1
                elif choice < self.proportions[7]:
                    # plain walking terrain
                    pass

                # Heightfield coordinate system
                self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

                robots_in_map = num_robots_per_map
                if j < left_over:
                    robots_in_map += 1

                env_origin_x = (i + 0.5) * self.env_length
                env_origin_y = (j + 0.5) * self.env_width
                x1 = int((self.env_length / 2. - 1) / self.horizontal_scale)
                x2 = int((self.env_length / 2. + 1) / self.horizontal_scale)
                y1 = int((self.env_width / 2. - 1) / self.horizontal_scale)
                y2 = int((self.env_width / 2. + 1) / self.horizontal_scale)
                env_origin_z = np.max(
                    terrain.height_field_raw[x1:x2,
                                             y1:y2]) * self.vertical_scale
                self.env_origins[i, j] = [
                    env_origin_x, env_origin_y, env_origin_z
                ]

        self.walkable_field_raw = ndimage.binary_dilation(self.walkable_field_raw, iterations=3).astype(int)

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, :2] = 0.
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


@torch.jit.script
def wrap_to_pi(angles):
    angles %= 2 * np.pi
    angles -= 2 * np.pi * (angles > np.pi)
    return angles

class PMBase(humanoid_amp_task.HumanoidAMPTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        self.config = cfg.env

        self.device = device_type + ':' + str(device_id)

        perturbations = self.config.get("perturbations", {})
        self.gravity_z = perturbations.get("gravity_z", -9.81)
        if "friction" in perturbations:
            self.config.simulator.plane.static_friction = perturbations["friction"]
            self.config.simulator.plane.dynamic_friction = perturbations["friction"]

        self.sensor_extent = cfg["env"].get("sensor_extent", 2)
        self.sensor_res = cfg["env"].get("sensor_res", 32)
        self.real_mesh = False
        self.load_humanoid_configs(cfg)
        self.load_smpl_configs(cfg)
        self.cfg = cfg
        self.num_envs = cfg["env"]["num_envs"]
        self.headless = cfg["headless"]
        self.power_reward = cfg["env"].get("power_reward", False)
        self.power_coefficient = cfg["env"].get("power_coefficient", 0.0005)
        self.fuzzy_target = cfg["env"].get("fuzzy_target", False)
        self.root_points = self.init_root_points()
        self.center_height_points = self.init_center_height_points()
        self.square_height_points = self.init_square_height_points()

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

        self.w_last = True

        self.create_terrain()
        self.build_termination_heights()
        self._failures = []
        self._distances = []
        self._current_accumulated_errors = (
                torch.zeros([self.num_envs], device=self.device, dtype=torch.float) - 1
        )
        self._current_failures = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.float
        )
        self._last_length = torch.zeros(
            [self.num_envs], device=self.device, dtype=torch.long
        )

        self.results = {}

    def _create_ground_plane(self):
        print("Creating ground plane")
        # import pdb;pdb.set_trace()
        if self.cfg.env.terrain is None:
            self.add_default_ground()
        else:
            self.create_training_ground()
        print("Ground plane created")

    def _compute_reset(self):
        time = self.progress_buf * self.dt
        env_ids = torch.arange(self.num_envs,
                               device=self.device,
                               dtype=torch.long)
        tar_pos = self._traj_gen.calc_pos(env_ids, time)
        ### ZL: entry point
        # self._traj_gen.update_sim_pos(self._humanoid_root_states[)

        root_states = self._humanoid_root_states
        center_height = self.get_center_heights(
            root_states, env_ids=None).mean(dim=-1, keepdim=True)

        # import ipdb
        # ipdb.set_trace()
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf, self.progress_buf, self._contact_forces,
            self._contact_body_ids, center_height, self._rigid_body_pos,
            tar_pos, self.max_episode_length, self._fail_dist,
            self._enable_early_termination, self._termination_heights, flags.no_collision_check)
        return
    def add_default_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction

        # plane_params.static_friction = 50
        # plane_params.dynamic_friction = 50

        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def create_training_ground(self):
        if self.cfg["env"].get("small_terrain", False):
            self.cfg["env"]["terrain"]['mapLength'] = 8
            self.cfg["env"]["terrain"]['mapWidth'] = 8

        self.terrain = Terrain(self.cfg["env"]["terrain"],
                               num_robots=self.num_envs,
                               device=self.device)

        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]
        tm_params.transform.p.x = 0
        tm_params.transform.p.y = 0
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg["env"]["terrain"]["staticFriction"]
        tm_params.dynamic_friction = self.cfg["env"]["terrain"]["dynamicFriction"]
        tm_params.restitution = self.cfg["env"]["terrain"]["restitution"]
        self.gym.add_triangle_mesh(self.sim,
                                   self.terrain.vertices.flatten(order='C'),
                                   self.terrain.triangles.flatten(order='C'),
                                   tm_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows,
                                                                            self.terrain.tot_cols).to(self.device)

    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = self.gravity_z
            return 2
        return 1

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        self.set_perturbations(env_ptr)

    def set_perturbations(self, env_ptr):
        perturbations = self.config.get("perturbations", {})
        if "friction" in perturbations:
            ground_friction = perturbations["friction"]
            foot_names = ["L_Ankle", "R_Ankle", "L_Toe", "R_Toe"]
            foot_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, 0, name) for name in foot_names]
            rb_shape = self.gym.get_actor_rigid_body_shape_indices(env_ptr, 0)
            rb_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, 0)
            for foot_handle in foot_handles:
                foot_shape = rb_shape[foot_handle]
                rb_shape_props[foot_shape.start].friction = ground_friction
                rb_shape_props[foot_shape.start].rolling_friction = ground_friction
                rb_shape_props[foot_shape.start].torsion_friction = ground_friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, 0, rb_shape_props)
        if "mass_multiplier" in perturbations:
            mass_multiplier = perturbations["mass_multiplier"]
            rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, 0)
            for body_name, multiplier in mass_multiplier.items():
                body_handle = self.gym.find_actor_rigid_body_handle(env_ptr, 0, body_name)
                rb_props[body_handle].mass *= multiplier
            self.gym.set_actor_rigid_body_properties(env_ptr, 0, rb_props)

    def _sample_ref_state(self, env_ids, vel_min=1, vel_range=0.5):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        # if (self._state_init == HumanoidAMP.StateInit.Random or self._state_init == HumanoidAMP.StateInit.Hybrid):
        #     motion_times = self._sample_time(motion_ids)
        # elif (self._state_init == HumanoidAMP.StateInit.Start):
        #     motion_times = torch.zeros(num_envs, device=self.device)
        # else:
        #     assert (
        #         False
        #     ), "Unsupported state initialization strategy: {:s}".format(
        #         str(self._state_init))
        motion_times = self._sample_time(motion_ids)

        if self.humanoid_type in ["smpl", "smplh", "smplx"]:
            curr_gender_betas = self.humanoid_shapes[env_ids]
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, body_vel, body_ang_vel = self._get_fixed_smpl_state_from_motionlib(
                motion_ids, motion_times, curr_gender_betas)
        else:
            root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel = self._motion_lib.get_motion_state(
                motion_ids, motion_times)
            rb_pos, rb_rot = None, None

        key_pos = rb_pos[:, self._key_body_ids]

        # if flags.random_heading:
        #     random_rot = np.zeros([num_envs, 3])
        #     random_rot[:, 2] = np.pi * (2 * np.random.random([num_envs]) - 1.0)
        #     random_heading_quat = torch.from_numpy(sRot.from_euler("xyz", random_rot).as_quat()).float().to(self.device)
        #     random_heading_quat_repeat = random_heading_quat[:, None].repeat(1, 24, 1)
        #     root_rot = quat_mul(random_heading_quat, root_rot).clone()
        #     rb_pos = quat_apply(random_heading_quat_repeat, rb_pos - root_pos[:, None, :]).clone() + root_pos[:, None, :]
        #     key_pos  = quat_apply(random_heading_quat_repeat[:, :4, :], (key_pos - root_pos[:, None, :])).clone() + root_pos[:, None, :]
        #     rb_rot = quat_mul(random_heading_quat_repeat, rb_rot).clone()
        #     root_ang_vel = quat_apply(random_heading_quat, root_ang_vel).clone()
        #     root_vel = quat_apply(random_heading_quat, root_vel).clone()

        return motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel
    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids, motion_times, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, body_vel, body_ang_vel = self._sample_ref_state(
            env_ids)
        ## Randomrized location setting
        new_root_xy = self.terrain.sample_valid_locations(self.num_envs, env_ids)
        # joblib.dump(self.terrain.sample_valid_locations(100000, torch.arange(100000)).detach().cpu(), "new_root_xy.pkl")
        # import ipdb; ipdb.set_trace()

        if flags.fixed:
            # new_root_xy[:, 0], new_root_xy[:, 1] = 0 , 0
            # new_root_xy[:, 0], new_root_xy[:, 1] = 134.8434 + env_ids , -28.9593
            # new_root_xy[:, 0], new_root_xy[:, 1] = 30 + env_ids * 4, 240
            new_root_xy[:, 0], new_root_xy[:, 1] = 84 + env_ids * 3, 143
            # new_root_xy[:, 0], new_root_xy[:, 1] = 95 + env_ids * 5, 307
            # new_root_xy[:, 0], new_root_xy[:, 1] = 27, 1 + env_ids * 2
            # x_grid, y_grid = torch.meshgrid(torch.arange(64), torch.arange(64))
            # new_root_xy[:, 0], new_root_xy[:, 1] = x_grid.flatten()[env_ids] * 2, y_grid.flatten()[env_ids] * 2
            # if env_ids[0] == 0:
            # new_root_xy[0, 0], new_root_xy[0, 1] = 34 , -81

        if flags.server_mode:
            new_traj = self._traj_gen.input_new_trajs(env_ids)
            new_root_xy[:, 0], new_root_xy[:, 1] = new_traj[:, 0, 0], new_traj[:, 0, 1]

        diff_xy = new_root_xy - root_pos[:, 0:2]
        root_pos[:, 0:2] = new_root_xy

        root_states = torch.cat([root_pos, root_rot], dim=1)

        center_height = self.get_center_heights(root_states, env_ids=env_ids).mean(dim=-1)

        if self.big_ankle:  # Big ankle needs a bit more room.
            center_height += 0.05

        root_pos[:, 2] += center_height
        key_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[..., 2] += center_height[:, None]
        rb_pos[..., 0:2] += diff_xy[:, None, :]
        key_pos[..., 2] += center_height[:, None]

        self._set_env_state(env_ids=env_ids,
                            root_pos=root_pos,
                            root_rot=root_rot,
                            dof_pos=dof_pos,
                            root_vel=root_vel,
                            root_ang_vel=root_ang_vel,
                            dof_vel=dof_vel,
                            rigid_body_pos=rb_pos,
                            rigid_body_rot=rb_rot,
                            rigid_body_vel=body_vel,
                            rigid_body_ang_vel=body_ang_vel,

                            )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        if flags.follow:
            self.start = True  ## Updating camera when reset

        return

    def get_center_heights(self, root_states, env_ids=None):
        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_center_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if self.humanoid_type in ["smpl", "smplh", "smplx"] and not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        if env_ids is None:
            points = quat_apply_yaw(
                base_quat.repeat(1, self.num_center_height_points,),
                self.center_height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(
                base_quat.repeat(1, self.num_center_height_points,),
                self.center_height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        heights = self.terrain.sample_height_points(points.clone(), env_ids=env_ids)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

    def init_center_height_points(self):
        # center_height_points
        y =  torch.tensor(np.linspace(-0.2, 0.2, 3),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.1, 0.1, 3),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_center_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_center_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_height_points(self):
        # 4mx4m square
        y =  torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent, self.sensor_res),device=self.device,requires_grad=False)
        x = torch.tensor(np.linspace(-self.sensor_extent, self.sensor_extent,
                                     self.sensor_res),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_square_fov_height_points(self):
        y = torch.tensor(np.linspace(-1, 1, 20),device=self.device,requires_grad=False)
        x =  torch.tensor(np.linspace(-0.02, 1.98, 20),device=self.device,requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_height_points,
                             3,
                            device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def init_root_points(self):
        y = torch.tensor(np.linspace(-0.5, 0.5, 20),
                         device=self.device,
                         requires_grad=False)
        x = torch.tensor(np.linspace(-0.25, 0.25, 10),
                         device=self.device,
                         requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_root_points = grid_x.numel()
        points = torch.zeros(self.num_envs,
                             self.num_root_points,
                             3,
                             device=self.device,
                             requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    def get_heights(self, root_states, env_ids=None):

        base_quat = root_states[:, 3:7]
        if self.cfg["env"]["terrain"]["terrainType"] == 'plane':
            return torch.zeros(self.num_envs,
                               self.num_height_points,
                               device=self.device,
                               requires_grad=False)
        elif self.cfg["env"]["terrain"]["terrainType"] == 'none':
            raise NameError("Can't measure height with terrain type 'none'")

        if self.humanoid_type in ["smpl", "smplh", "smplx"] and not self._has_upright_start:
            base_quat = remove_base_rot(base_quat)

        heading_rot = torch_utils.calc_heading_quat(base_quat, w_last=True)

        if env_ids is None:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points) + (root_states[:, :3]).unsqueeze(1)
        else:
            points = quat_apply(
                heading_rot.repeat(1, self.num_height_points).reshape(-1, 4),
                self.height_points[env_ids]) + (
                    root_states[:, :3]).unsqueeze(1)

        if self.velocity_map:
            root_states_all = self._humanoid_root_states
        else:
            root_states_all = None

        if (self._divide_group or flags.divide_group) and not self._group_obs and not self._disable_group_obs:
            heading_rot_all = torch_utils.calc_heading_quat(self._humanoid_root_states[:, 3:7])
            root_points = quat_apply(
                heading_rot_all.repeat(1, self.num_root_points).reshape(-1, 4),
                self.root_points) + (self._humanoid_root_states[:, :3]).unsqueeze(1)
            # update heights with root points
            heights = self.terrain.sample_height_points(
                points.clone(),
                root_states = root_states_all,
                root_points = root_points,
                env_ids=env_ids,
                num_group_people=self._group_num_people,
                group_ids = self._group_ids)
        else:
            heights = self.terrain.sample_height_points(
                points.clone(),
                root_states=root_states_all,
                root_points=None,
                env_ids=env_ids,
            )
        # heights = self.terrain.sample_height_points(points.clone(), None)
        num_envs = self.num_envs if env_ids is None else len(env_ids)

        return heights.view(num_envs, -1)

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

    def accumulate_errors(self):
        self.results["reach_success"] = 1.0 - torch.Tensor(self._failures).mean()
        self.results["reach_distance"] = torch.Tensor(self._distances).mean()

    def create_terrain(self):
        # self.terrain: Terrain = instantiate(
        #     self.config.terrain,
        #     scene_lib=None,
        #     num_envs=self.num_envs,
        #     device=self.device,
        # )

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

@torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf,
                           contact_body_ids, center_height, rigid_body_pos,
                           tar_pos, max_episode_length, fail_dist,
                           enable_early_termination, termination_heights, disableCollision):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, bool, Tensor, bool) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        ## torch.sum to disable self-collision.
        # force_threshold = 200
        force_threshold = 50
        body_contact_force = torch.sqrt(torch.square(torch.abs(masked_contact_buf.sum(dim=-2))).sum(dim=-1)) > force_threshold

        # has_fallen = torch.logical_and(body_contact_force, fall_height)
        has_fallen = body_contact_force
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)

        root_pos = rigid_body_pos[..., 0, :]
        tar_delta = tar_pos[..., 0:2] - root_pos[...,0:2]  # also reset if toooo far away from the target trajectory
        tar_dist_sq = torch.sum(tar_delta * tar_delta, dim=-1)
        tar_fail = tar_dist_sq > fail_dist * fail_dist

        has_failed = torch.logical_or(has_fallen, tar_fail)
        # if has_fallen.any():
        #     import ipdb
        #     ipdb.set_trace()

        if disableCollision:
            has_failed[:] = False

        ############################## Debug ##############################
        # if torch.sum(has_fallen) > 0:
        #     import ipdb; ipdb.set_trace()
        #     print("???")
        # mujoco_joint_names = np.array(['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand'])
        # print( mujoco_joint_names[masked_contact_buf[0, :, 0].nonzero().cpu().numpy()])
        ############################## Debug ##############################


        # has_failed[:] = False

        terminated = torch.where(has_failed, torch.ones_like(reset_buf), terminated)


        # if torch.sum(terminated) > 0:
        #     termianted_progress = progress_buf[torch.where(terminated)]
        #     print(torch.where(termianted_progress < 30), termianted_progress[termianted_progress < 30])

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated