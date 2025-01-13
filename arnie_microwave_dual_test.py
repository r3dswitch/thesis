from unittest import TextTestRunner
from matplotlib.pyplot import axis
from PIL import Image as Im

import numpy as np
import os
import random
import torch
import math

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

def iprint(*strings):
    print(strings)
    exit()

global min 
min = 1000
global max 
max = -1
# import wandb
# wandb.init()

class ArnieMicrowaveDual(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = False
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = 1

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.diana_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        self.ignore_z = False

        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": 417 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 417 + self.num_point_cloud_feature_dim * 3,
            "full_state": 417
        }
        self.num_hand_obs = 199 
        self.up_axis = 'z'

        self.left_fingertips = ["Left_Index_Phadist", "Left_Middle_Phadist", "Left_Ring_Phadist", "Left_Little_Phadist", "Left_Thumb_Phadist"]
        self.right_fingertips = ["Right_Index_Phadist", "Right_Middle_Phadist", "Right_Ring_Phadist", "Right_Little_Phadist", "Right_Thumb_Phadist"]

        self.num_fingertips = 10

        self.base_idx = 0
        self.hand_idx = 7
        self.index_idx = 11
        self.little_idx = 15
        self.middle_idx = 19
        self.ring_idx = 23
        self.thumb_idx = 27
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = False

        num_states = 0
    
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 27
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 54

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.left_jacobian = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "left_hand"))
        self.left_j_eef = self.l_jacobian[:, 7 - 1, :, :7]
        print(self.left_j_eef)

        self.right_jacobian = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "right_hand"))
        self.right_j_eef = self.r_jacobian[:, 7 - 1, :, :7]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.left_diana_default_dof_pos = to_torch([-1.66, #-0.99, #-1.02,
                -1.35, #-1.11,
                -1.69, #-3.12,
                1.66, #1.13,
                -3.00, #-2.33,
                0.11, #0.38,
                -1.47, #-1.60,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        dtype=torch.float, device=self.device)
        self.right_diana_default_dof_pos = to_torch([1.00,
                -1.22,
                3.12,
                1.16,
                1.56,
                0.30,
                3.12,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.left_diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :27]
        self.left_diana_dof_pos = self.left_diana_dof_state[..., 0]
        self.left_diana_dof_vel = self.left_diana_dof_state[..., 1]

        self.right_diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, 27:54]
        self.right_diana_dof_pos = self.right_diana_dof_state[..., 0]
        self.right_diana_dof_vel = self.right_diana_dof_state[..., 1]

        self.left_object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, 54:]
        self.left_object_dof_pos = self.left_object_dof_state[..., 0]
        self.left_object_dof_vel = self.left_object_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
       
        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets"
        
        left_diana_asset_file = "urdf/left_hit_more_friction.urdf"
        right_diana_asset_file = "urdf/right_hit.urdf"
        
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        left_object_asset_file = "urdf/microwave/microwave_relabel.urdf"
        right_object_asset_file = "urdf/cube.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
         
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS # if set the drive mode with None, may lead to unstable robot movement)
        asset_options.use_mesh_materials = True
        asset_options.thickness = 0.01

        left_diana_asset = self.gym.load_asset(self.sim, asset_root, left_diana_asset_file, asset_options)
        right_diana_asset = self.gym.load_asset(self.sim, asset_root, right_diana_asset_file, asset_options)

        self.num_diana_bodies = self.gym.get_asset_rigid_body_count(left_diana_asset)
        self.num_diana_shapes = self.gym.get_asset_rigid_shape_count(left_diana_asset)
        self.num_diana_dofs = self.gym.get_asset_dof_count(left_diana_asset)
        
        diana_dof_props = self.gym.get_asset_dof_properties(left_diana_asset)
        self.num_arm_hand_actuators = self.num_diana_dofs
        self.actuated_dof_indices = [i for i in range(7, self.num_diana_dofs)]
        
        self.diana_dof_lower_limits = []
        self.diana_dof_upper_limits = []
        self.diana_dof_default_pos = []
        self.diana_dof_default_vel = []
        
        robot_lower_qpos = []
        robot_upper_qpos = []

        for i in range(self.num_diana_dofs):
            diana_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i<7:
                diana_dof_props['stiffness'][i] = 400
                diana_dof_props['effort'][i] = 200
                diana_dof_props['damping'][i] = 80
            else:
                diana_dof_props['velocity'][i] = 10.0
                diana_dof_props['stiffness'][i] = 20.0
                diana_dof_props['effort'][i] = 0.7
                diana_dof_props['damping'][i] = 1.0
            
            self.diana_dof_lower_limits.append(diana_dof_props['lower'][i])
            self.diana_dof_upper_limits.append(diana_dof_props['upper'][i])
            self.diana_dof_default_pos.append(0.0)
            self.diana_dof_default_vel.append(0.0)

        self.diana_dof_lower_limits = to_torch(self.diana_dof_lower_limits, device=self.device)
        self.diana_dof_upper_limits = to_torch(self.diana_dof_upper_limits, device=self.device)
        
        self.diana_dof_lower_qvel = to_torch(-diana_dof_props["velocity"], device=self.device)
        self.diana_dof_upper_qvel = to_torch(diana_dof_props["velocity"], device=self.device)

        self.diana_dof_default_pos = to_torch(self.diana_dof_default_pos, device=self.device)
        self.diana_dof_default_vel = to_torch(self.diana_dof_default_vel, device=self.device)

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.use_mesh_materials = True
        asset_options.thickness = 1 # to prevent collisions
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        left_object_asset = self.gym.load_asset(self.sim, asset_root, left_object_asset_file, object_asset_options)
        object_asset_options.fix_base_link = False
        right_object_asset = self.gym.load_asset(self.sim, asset_root, right_object_asset_file, object_asset_options)

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(left_object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(left_object_asset)

        self.num_object_dofs = self.gym.get_asset_dof_count(left_object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(left_object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)

        table_dims = gymapi.Vec3(0.75, 1.5, 0.55)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())

        left_diana_start_pose = gymapi.Transform()
        left_diana_start_pose.p = gymapi.Vec3(0, -0.1, 1.45)
        left_diana_start_pose.r = gymapi.Quat().from_euler_zyx(1.5652925671162337, 0, 0.227)

        right_diana_start_pose = gymapi.Transform()
        right_diana_start_pose.p = gymapi.Vec3(0, 0.1, 1.45) 
        right_diana_start_pose.r = gymapi.Quat().from_euler_zyx(-1.5652925671162337, 0, -0.227)

        left_object_start_pose = gymapi.Transform()
        left_object_start_pose.p = gymapi.Vec3(-0.85, -0.6, 0.8)
        left_object_start_pose.r = gymapi.Quat().from_euler_zyx(3.141592, 3.141592, 0)

        right_object_start_pose = gymapi.Transform()
        right_object_start_pose.p = gymapi.Vec3(-0.85, 0.2, 0.8)
        right_object_start_pose.r = gymapi.Quat().from_euler_zyx(3.141592, 3.141592, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(-0.85, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        max_agg_bodies = self.num_diana_bodies * 2 + 3 * self.num_object_bodies + 1
        max_agg_shapes = self.num_diana_shapes * 2 + 3 * self.num_object_shapes + 1

        self.dianas = []
        self.envs = []

        self.left_object_init_state = []
        self.right_object_init_state = []
        
        self.left_hand_init_state = []
        self.right_hand_init_state = []

        self.left_hand_indices = []
        self.right_hand_indices = []

        self.left_object_indices = []
        self.right_object_indices = []
        self.table_indices = []
        
        self.left_fingertip_handles = [self.gym.find_asset_rigid_body_index(left_diana_asset, name) for name in self.left_fingertips]
        self.right_fingertip_handles = [self.gym.find_asset_rigid_body_index(right_diana_asset, name) for name in self.right_fingertips]

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            left_diana_actor = self.gym.create_actor(env_ptr, left_diana_asset, left_diana_start_pose, "left_hand", i, -1, 0)
            right_diana_actor = self.gym.create_actor(env_ptr, right_diana_asset, right_diana_start_pose, "right_hand", i, 0, 0)
            
            self.left_hand_init_state.append([left_diana_start_pose.p.x, left_diana_start_pose.p.y, left_diana_start_pose.p.z,
                                           left_diana_start_pose.r.x, left_diana_start_pose.r.y, left_diana_start_pose.r.z, left_diana_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.right_hand_init_state.append([right_diana_start_pose.p.x, right_diana_start_pose.p.y, right_diana_start_pose.p.z,
                                           right_diana_start_pose.r.x, right_diana_start_pose.r.y, right_diana_start_pose.r.z, right_diana_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, left_diana_actor, diana_dof_props)
            
            left_hand_idx = self.gym.get_actor_index(env_ptr, left_diana_actor, gymapi.DOMAIN_SIM)
            self.left_hand_indices.append(left_hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, right_diana_actor, diana_dof_props)
            right_hand_idx = self.gym.get_actor_index(env_ptr, right_diana_actor, gymapi.DOMAIN_SIM)
            self.right_hand_indices.append(right_hand_idx)            
            
            left_object_handle = self.gym.create_actor(env_ptr, left_object_asset, left_object_start_pose, "left_object", i, 0, 0)
            self.gym.set_actor_scale(env_ptr, left_object_handle, 0.4)
            self.left_object_init_state.append([left_object_start_pose.p.x, left_object_start_pose.p.y, left_object_start_pose.p.z,
                                           left_object_start_pose.r.x, left_object_start_pose.r.y, left_object_start_pose.r.z, left_object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            left_object_idx = self.gym.get_actor_index(env_ptr, left_object_handle, gymapi.DOMAIN_SIM)
            self.left_object_indices.append(left_object_idx)
            
            right_object_handle = self.gym.create_actor(env_ptr, right_object_asset, right_object_start_pose, "right_object", i, 0, 0)
            self.right_object_init_state.append([right_object_start_pose.p.x, right_object_start_pose.p.y, right_object_start_pose.p.z,
                                           right_object_start_pose.r.x, right_object_start_pose.r.y, right_object_start_pose.r.z, right_object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_scale(env_ptr, right_object_handle, 2.0)
            right_object_idx = self.gym.get_actor_index(env_ptr, right_object_handle, gymapi.DOMAIN_SIM)
            self.right_object_indices.append(right_object_idx)

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.dianas.append(left_diana_actor)
            self.dianas.append(right_diana_actor)

        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 256
        self.camera_props.height = 256
        self.camera_props.enable_tensors = True
        self.camera_handle = self.gym.create_camera_sensor(self.envs[0], self.camera_props)
        self.gym.set_camera_location(self.camera_handle, self.envs[0], gymapi.Vec3(0.25, -0., 2), gymapi.Vec3(-0.24, -0., 0))
        
        self.left_object_init_state = to_torch(self.left_object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.right_object_init_state = to_torch(self.right_object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        
        self.left_hand_init_state = to_torch(self.left_hand_init_state, device=self.device).view(self.num_envs, 13)
        self.right_hand_init_state = to_torch(self.right_hand_init_state, device=self.device).view(self.num_envs, 13)
        
        self.left_fingertip_handles = self.right_fingertip_handles = to_torch(self.right_fingertip_handles, dtype=torch.long, device=self.device)
        
        self.left_hand_indices = to_torch(self.left_hand_indices, dtype=torch.long, device=self.device) # 0
        self.right_hand_indices = to_torch(self.right_hand_indices, dtype=torch.long, device=self.device) # 1
        
        self.left_object_indices = to_torch(self.left_object_indices, dtype=torch.long, device=self.device) # 2
        self.right_object_indices = to_torch(self.right_object_indices, dtype=torch.long, device=self.device) # 3
        
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device) # 4
        
        self.right_object_pos = self.right_object_init_state[:,:3]
        self.right_object_rot = self.right_object_init_state[:,3:7]
        self.microwave_handle_pos = self.left_object_init_state[:,:3]
        self.microwave_handle_rot = self.left_object_init_state[:,3:7]
        
        self.left_hand_pos = self.left_hand_init_state[:,:3]
        self.left_hand_rot = self.left_hand_init_state[:,3:7]
        self.right_hand_pos = self.right_hand_init_state[:,:3]
        self.right_hand_rot = self.right_hand_init_state[:,3:7]
        
        self.init_data()
    
    def init_data(self):
        left_hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.dianas[0], "link_7")
        right_hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.dianas[1], "link_7")
        
        left_hand_pose = self.gym.get_rigid_transform(self.envs[0], left_hand)
        right_hand_pose = self.gym.get_rigid_transform(self.envs[0], right_hand)

        left_bot_hand_pos = to_torch([left_hand_pose.p.x, left_hand_pose.p.y, left_hand_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        left_bot_hand_rot = to_torch([left_hand_pose.r.x, left_hand_pose.r.y, left_hand_pose.r.z, left_hand_pose.r.w], device=self.device).repeat((self.num_envs, 1))
        right_bot_hand_pos = to_torch([right_hand_pose.p.x, right_hand_pose.p.y, right_hand_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        right_bot_hand_rot = to_torch([right_hand_pose.r.x, right_hand_pose.r.y, right_hand_pose.r.z, right_hand_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.left_hand_pos = torch.zeros_like(left_bot_hand_pos)
        self.left_hand_rot = torch.zeros_like(left_bot_hand_rot)
        self.right_hand_pos = torch.zeros_like(right_bot_hand_pos)
        self.right_hand_rot = torch.zeros_like(right_bot_hand_rot)
    
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.right_object_pos, self.right_object_pos, self.microwave_handle_pos, self.microwave_handle_rot,
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, False
        )
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        # self.gym.clear_lines(self.viewer)
        # self.gym.refresh_rigid_body_state_tensor(self.sim)

        # for i in range(self.num_envs):
        #     self.add_debug_lines(self.envs[i], self.microwave_handle_pos[i], self.microwave_handle_rot[i])
        
        # wandb.log({"reward:" : torch.mean(self.rew_buf)})

        
        camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.camera_handle, gymapi.IMAGE_COLOR)
        torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
        camera_image = torch_rgba_tensor.cpu().numpy()
        camera_image = Im.fromarray(camera_image)
        
        # img = wandb.Image(camera_image, caption="camera view")
        # wandb.log({"Image:" : img})

    def compute_observations(self):
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.right_object_pose = self.root_state_tensor[self.right_object_indices, 0:7]
        self.right_object_pos = self.root_state_tensor[self.right_object_indices, 0:3]
        self.right_object_rot = self.root_state_tensor[self.right_object_indices, 3:7]
        
        self.microwave_handle_pos = self.rigid_body_states[:, self.num_diana_bodies * 2 + 2, 0:3] # 1: base, 2: link1, 3: link0
        self.microwave_handle_rot = self.rigid_body_states[:, self.num_diana_bodies * 2 + 2, 3:7]
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.2)
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.4)
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.05) 

        self.left_hand_pos = self.rigid_body_states[:, self.hand_idx, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, self.hand_idx, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)
        
        self.right_hand_pos = self.rigid_body_states[:, self.hand_idx + self.num_diana_bodies, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, self.hand_idx + self.num_diana_bodies, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)
    
        self.left_hand_ff_pos = self.rigid_body_states[:, self.index_idx, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, self.index_idx, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, self.middle_idx, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, self.middle_idx, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, self.ring_idx, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, self.ring_idx, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, self.little_idx, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, self.little_idx, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, self.thumb_idx, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, self.thumb_idx, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.right_hand_ff_pos = self.rigid_body_states[:, self.index_idx + self.num_diana_bodies, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, self.index_idx + self.num_diana_bodies, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, self.middle_idx + self.num_diana_bodies, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, self.middle_idx + self.num_diana_bodies, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, self.ring_idx + self.num_diana_bodies, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, self.ring_idx + self.num_diana_bodies, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, self.little_idx + self.num_diana_bodies, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, self.little_idx + self.num_diana_bodies, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, self.thumb_idx + self.num_diana_bodies, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, self.thumb_idx + self.num_diana_bodies, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.left_fingertip_state = self.rigid_body_states[:, self.left_fingertip_handles][:, :, 0:13]
        self.left_fingertip_pos = self.rigid_body_states[:, self.left_fingertip_handles][:, :, 0:3]

        self.right_fingertip_state = self.rigid_body_states[:, self.right_fingertip_handles][:, :, 0:13]
        self.right_fingertip_pos = self.rigid_body_states[:, self.right_fingertip_handles][:, :, 0:3]

        if self.obs_type == "full_state":
            self.compute_full_state()

    def compute_full_state(self, asymm_obs=False):
        self.obs_buf[:,:27] = unscale(self.left_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:,27:54] = unscale(self.right_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, 54:108] = self.actions
        self.obs_buf[:,108:111] = self.left_hand_pos
        self.obs_buf[:,111:115] = self.left_hand_rot
        self.obs_buf[:,115:118] = self.right_hand_pos
        self.obs_buf[:,118:122] = self.right_hand_rot
        self.obs_buf[:,122:187] = self.left_fingertip_state.reshape(self.num_envs, 65)
        self.obs_buf[:,187:252] = self.left_fingertip_state.reshape(self.num_envs, 65)
        self.gym.end_access_image_tensors(self.sim)

    def reset(self, env_ids, goal_env_ids):
        # print("resets")
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.left_object_dof_state[env_ids, :] = torch.zeros_like(self.left_object_dof_state[env_ids])

        object_indices = torch.unique(self.left_object_indices[env_ids]).to(torch.int32)

        pos_left = self.left_diana_default_dof_pos[:7]
        pos_right = self.right_diana_default_dof_pos[:7]

        self.left_diana_dof_pos[env_ids, :7] = pos_left
        self.right_diana_dof_pos[env_ids, :7] = pos_right

        self.prev_targets[env_ids, :7] = pos_left
        self.cur_targets[env_ids, :7] = pos_left

        self.prev_targets[env_ids, 27:34] = pos_right
        self.cur_targets[env_ids, 27:34] = pos_right

        left_hand_indices = self.left_hand_indices[env_ids].to(torch.int32)
        right_hand_indices = self.right_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([left_hand_indices, right_hand_indices]).to(torch.int32))
        all_indices = torch.unique(torch.cat([all_hand_indices, object_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_indices), len(all_indices))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if True: #self.progress_buf[0] < 125: 
            # pos_err = self.actions[:, 0:3] * 0.2
            # rot_err = self.actions[:, 3:7] * 0.05
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            

            t_rot =  to_torch([0, 0, 0], device=self.device).repeat((self.num_envs, 1))
            target_rot = quat_from_euler_xyz(t_rot[:,0],t_rot[:,1],t_rot[:,2])
            rot_err = orientation_error(target_rot, self.left_hand_rot.clone())

            t_pos = to_torch([-1, -1, 1], device=self.device).repeat((self.num_envs, 1))# self.microwave_handle_pos.clone()
            pos_err = t_pos - self.left_hand_pos
            # pos_err[:,0] -= 0.5
            # pos_err[:,1] -= 0.2
            # pos_err[:,2] -= 0.05
            pos_err += self.actions[:, 0:3] * 0.05

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], t_pos[i], target_rot[i])

            dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
            delta = control_ik(self.left_j_eef, self.device, dpose, self.num_envs)

            self.cur_targets[:, :7] = self.left_diana_dof_pos[:, 0:7] + delta[:, :7]
        else: 
            self.cur_targets[:, 7:27] = scale(self.actions[:, 7:27], self.diana_dof_lower_limits[7:27], self.diana_dof_upper_limits[7:27])

        self.cur_targets[:,:27] = tensor_clamp(self.cur_targets[:,:27], self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.cur_targets[:,27:54] = torch.zeros_like(self.cur_targets[:,27:54])
        self.prev_targets = self.cur_targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        # wandb.log({"actions:" : self.actions[:,0]})
        # wandb.log({"pos_err:" : pos_err[:,0]})
        # wandb.log({"rot_err:" : rot_err[:,0]})
        # wandb.log({"dpose:" : dpose[:,0]})
        # wandb.log({"delta:" : delta[:,0]})
        # wandb.log({"cur_targets:" : self.cur_targets[:,0]})

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
       
    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, right_object_pos, right_object_rot, microwave_handle_pos, microwave_handle_rot,
    left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    
    # global max
    # global min

    left_hand_finger_dist = (torch.norm(microwave_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(microwave_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(microwave_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(microwave_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(microwave_handle_pos - left_hand_th_pos, p=2, dim=-1))
    
    right_hand_finger_dist = (torch.norm(right_object_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(right_object_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(right_object_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(right_object_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(right_object_pos - right_hand_th_pos, p=2, dim=-1))
    
    left_hand_dist_rew = left_hand_finger_dist
    right_hand_dist_rew = right_hand_finger_dist

    open_rew = (microwave_handle_pos[:,0] + 0.75) * 10

    reward = torch.exp(-0.1 * left_hand_dist_rew) # + open_rew - action_penalty * action_penalty_scale
    # print(left_hand_dist_rew, reward)
    # dist = torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale)) * 10
    
    # if(dist < min):
    #     min = dist
    # elif(dist >= max):
    #     max = dist
        
    # # print(min, max)
    # print(left_hand_finger_dist)
    # resets = torch.where(left_hand_dist_rew <= 0.5, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(left_hand_dist_rew >= 10, torch.ones_like(reset_buf), reset_buf) #distance termination condition
    # resets = torch.where(left_hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf)
    
    successes = torch.where(successes == 0, 
                            torch.where(open_rew > 3.5, 
                            torch.ones_like(successes), successes), successes)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()
    
    return reward, resets, goal_resets, progress_buf, successes, cons_successes

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):
    damping = 0.05
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u