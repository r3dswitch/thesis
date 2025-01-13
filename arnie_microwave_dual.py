from PIL import Image as Im

import numpy as np
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

def iprint(*strings):
    print(strings)
    exit()

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
        self.reset_dof_pos_noise = 0.01
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.diana_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = 0.8 #self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)

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

        self.num_hand_obs = 200
        self.up_axis = 'z'

        self.left_fingertips = ["Left_Index_Phadist", "Left_Middle_Phadist", "Left_Ring_Phadist", "Left_Little_Phadist", "Left_Thumb_Phadist"]
        self.right_fingertips = ["Right_Index_Phadist", "Right_Middle_Phadist", "Right_Ring_Phadist", "Right_Little_Phadist", "Right_Thumb_Phadist"]

        self.num_fingertips = 10

        self.fingertip_obs = True

        num_states = 0

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 13 #27
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 26 #54

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        if self.obs_type in ["point_cloud"]:
            from PIL import Image as Im
            from bidexhands.utils import o3dviewer

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.8, 0.2, 1.3)
            cam_target = gymapi.Vec3(-0.1, 0.2, 0.6)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_idx = 0
        self.hand_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.left_hand_indices[0], "link_7", gymapi.DOMAIN_ENV)
        self.index_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.left_hand_indices[0], "Left_Index_Phadist", gymapi.DOMAIN_ENV)
        self.little_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.left_hand_indices[0], "Left_Little_Phadist", gymapi.DOMAIN_ENV)
        self.middle_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.left_hand_indices[0], "Left_Middle_Phadist", gymapi.DOMAIN_ENV)
        self.ring_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.left_hand_indices[0], "Left_Ring_Phadist", gymapi.DOMAIN_ENV)
        self.thumb_idx = self.gym.find_actor_rigid_body_index(self.envs[0], self.left_hand_indices[0], "Left_Thumb_Phadist", gymapi.DOMAIN_ENV)

        self.left_diana_default_dof_pos = to_torch([1.56,0.78,0.67,1.88,0.76,-0.83,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=torch.float, device=self.device)
        self.right_diana_default_dof_pos = to_torch([0.24,-0.70,-2.17,1.90,-2.04,-1.35,0.64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],dtype=torch.float, device=self.device)
        self.microwave_default_dof_pos = torch.zeros(self.num_object_dofs, dtype=torch.float, device=self.device)
        
        self.left_arm_dof_trajectory_list = []
        self.left_arm_dof_trajectory = to_torch([1.87,-0.03,-0.03,2.08,1.76,-1.18,1.05, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float, device=self.device)
        self.left_arm_dof_trajectory_list.append(self.left_arm_dof_trajectory)
        self.left_arm_dof_trajectory = to_torch([0.24,-1.11,-2.27,1.24,-1.95,-0.95,0.35, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float, device=self.device)
        self.left_arm_dof_trajectory_list.append(self.left_arm_dof_trajectory)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.left_diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_diana_dofs]
        self.left_diana_dof_pos = self.left_diana_dof_state[..., 0]
        self.left_diana_dof_vel = self.left_diana_dof_state[..., 1]

        self.right_diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_diana_dofs:self.num_diana_dofs*2]
        self.right_diana_dof_pos = self.right_diana_dof_state[..., 0]
        self.right_diana_dof_vel = self.right_diana_dof_state[..., 1]

        self.microwave_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, 54:]
        self.microwave_dof_pos = self.microwave_dof_state[..., 0]
        self.microwave_dof_vel = self.microwave_dof_state[..., 1]
        
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

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
        left_diana_asset_file = "urdf/left_hit.urdf"
        right_diana_asset_file = "urdf/right_hit.urdf"
        
        microwave_asset_file = "urdf/microwave/microwave_relabel.urdf"
        cube_asset_file = "urdf/cube.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
         
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS # if set the drive mode with None, may lead to unstable robot movement)
        asset_options.thickness = 0.01

        left_diana_asset = self.gym.load_asset(self.sim, asset_root, left_diana_asset_file, asset_options)
        right_diana_asset = self.gym.load_asset(self.sim, asset_root, right_diana_asset_file, asset_options)

        self.num_diana_bodies = self.gym.get_asset_rigid_body_count(left_diana_asset)
        self.num_diana_shapes = self.gym.get_asset_rigid_shape_count(left_diana_asset)
        self.num_diana_dofs = self.gym.get_asset_dof_count(left_diana_asset)
        
        diana_dof_props = self.gym.get_asset_dof_properties(left_diana_asset)
        
        self.diana_dof_lower_limits = []
        self.diana_dof_upper_limits = []
        
        for i in range(self.num_diana_dofs):
            self.diana_dof_lower_limits.append(diana_dof_props['lower'][i])
            self.diana_dof_upper_limits.append(diana_dof_props['upper'][i])

        self.diana_dof_lower_limits = to_torch(self.diana_dof_lower_limits, device=self.device)
        self.diana_dof_upper_limits = to_torch(self.diana_dof_upper_limits, device=self.device)
        
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link = True
        object_asset_options.use_mesh_materials = True
        asset_options.thickness = 1 # to prevent collisions
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        microwave_asset = self.gym.load_asset(self.sim, asset_root, microwave_asset_file, object_asset_options)
        object_asset_options.fix_base_link = False
        cube_asset = self.gym.load_asset(self.sim, asset_root, cube_asset_file, object_asset_options)

        self.num_object_bodies = self.gym.get_asset_rigid_body_count(microwave_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(microwave_asset)

        self.num_object_dofs = self.gym.get_asset_dof_count(microwave_asset)
        object_dof_props = self.gym.get_asset_dof_properties(microwave_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)

        table_dims = gymapi.Vec3(1, 2, 0.55)
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

        microwave_start_pose = gymapi.Transform()
        microwave_start_pose.p = gymapi.Vec3(-0.8, -0.5, 0.75)
        microwave_start_pose.r = gymapi.Quat().from_euler_zyx(3.141592, 3.141592, 0)

        cube_start_pose = gymapi.Transform()
        cube_start_pose.p = gymapi.Vec3(-0.5, 0.4, 0.575)
        cube_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(-0.8, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        max_agg_bodies = self.num_diana_bodies * 2 + 20
        max_agg_shapes = self.num_diana_shapes * 2 + 20

        self.dianas = []
        self.envs = []

        self.microwave_init_state = []
        self.cube_init_state = []
        
        self.left_hand_init_state = []
        self.right_hand_init_state = []

        self.left_hand_indices = []
        self.right_hand_indices = []

        self.microwave_indices = []
        self.cube_indices = []
        self.table_indices = []
        
        self.left_fingertip_handles = [self.gym.find_asset_rigid_body_index(left_diana_asset, name) for name in self.left_fingertips]
        self.right_fingertip_handles = [self.gym.find_asset_rigid_body_index(right_diana_asset, name) for name in self.right_fingertips]

        if self.obs_type in ["point_cloud"]:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 768
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

            if self.point_cloud_debug:
                import open3d as o3d
                from bidexhands.utils.o3dviewer import PointcloudVisualizer
                self.pointCloudVisualizer = PointcloudVisualizer()
                self.pointCloudVisualizerInitialized = False
                self.o3d_pc = o3d.geometry.PointCloud()
            else :
                self.pointCloudVisualizer = None

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
            
            microwave_handle = self.gym.create_actor(env_ptr, microwave_asset, microwave_start_pose, "microwave", i, 0, 0)
            self.gym.set_actor_scale(env_ptr, microwave_handle, 0.45)
            self.microwave_init_state.append([microwave_start_pose.p.x, microwave_start_pose.p.y, microwave_start_pose.p.z,
                                           microwave_start_pose.r.x, microwave_start_pose.r.y, microwave_start_pose.r.z, microwave_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            microwave_idx = self.gym.get_actor_index(env_ptr, microwave_handle, gymapi.DOMAIN_SIM)
            self.microwave_indices.append(microwave_idx)
            
            cube_handle = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 0, 0)
            self.cube_init_state.append([cube_start_pose.p.x, cube_start_pose.p.y, cube_start_pose.p.z,
                                           cube_start_pose.r.x, cube_start_pose.r.y, cube_start_pose.r.z, cube_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_scale(env_ptr, cube_handle, 1.0)
            cube_idx = self.gym.get_actor_index(env_ptr, cube_handle, gymapi.DOMAIN_SIM)
            self.cube_indices.append(cube_idx)

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            
            if self.obs_type in ["point_cloud"]:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.25, -0., 1.0), gymapi.Vec3(-0.24, -0., 0))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.dianas.append(left_diana_actor)
            self.dianas.append(right_diana_actor)

        self.microwave_init_state = to_torch(self.microwave_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.cube_init_state = to_torch(self.cube_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        
        self.left_hand_init_state = to_torch(self.left_hand_init_state, device=self.device).view(self.num_envs, 13)
        self.right_hand_init_state = to_torch(self.right_hand_init_state, device=self.device).view(self.num_envs, 13)
        
        self.left_fingertip_handles = self.right_fingertip_handles = to_torch(self.right_fingertip_handles, dtype=torch.long, device=self.device)
        
        self.left_hand_indices = to_torch(self.left_hand_indices, dtype=torch.long, device=self.device) # 0
        self.right_hand_indices = to_torch(self.right_hand_indices, dtype=torch.long, device=self.device) # 1
        
        self.microwave_indices = to_torch(self.microwave_indices, dtype=torch.long, device=self.device) # 2
        self.cube_indices = to_torch(self.cube_indices, dtype=torch.long, device=self.device) # 3
        
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device) # 4
        
        self.cube_pos = self.cube_init_state[:,:3]
        self.cube_rot = self.cube_init_state[:,3:7]
        self.microwave_handle_pos = self.microwave_init_state[:,:3]
        self.microwave_handle_rot = self.microwave_init_state[:,3:7]
        
        self.left_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "left_hand")
        self.l_jacobian = gymtorch.wrap_tensor(self.left_jacobian)
        
        self.right_jacobian = self.gym.acquire_jacobian_tensor(self.sim, "right_hand")
        self.r_jacobian = gymtorch.wrap_tensor(self.right_jacobian)

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
            self.max_episode_length, self.cube_pos, self.cube_pos, self.microwave_handle_pos, self.microwave_handle_rot,
            self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.actions, self.action_penalty_scale
        )
        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes

    def compute_observations(self):
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.cube_pose = self.root_state_tensor[self.cube_indices, 0:7]
        self.cube_pos = self.root_state_tensor[self.cube_indices, 0:3]
        self.cube_rot = self.root_state_tensor[self.cube_indices, 3:7]
        
        self.microwave_handle_pos = self.rigid_body_states[:, self.num_diana_bodies * 2 + 2, 0:3] # 1: base, 2: link1, 3: link0
        self.microwave_handle_rot = self.rigid_body_states[:, self.num_diana_bodies * 2 + 2, 3:7]
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.2)
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.4)
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.075)

        self.left_hand_pos = self.rigid_body_states[:, self.hand_idx, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, self.hand_idx, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0)
        
        self.right_hand_pos = self.rigid_body_states[:, self.hand_idx + self.num_diana_bodies, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, self.hand_idx + self.num_diana_bodies, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0)

        self.left_hand_ff_pos = self.rigid_body_states[:, self.index_idx, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, self.index_idx, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, self.middle_idx, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, self.middle_idx, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, self.ring_idx, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, self.ring_idx, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, self.little_idx, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, self.little_idx, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, self.thumb_idx, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, self.thumb_idx, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.right_hand_ff_pos = self.rigid_body_states[:, self.index_idx + self.num_diana_bodies, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, self.index_idx + self.num_diana_bodies, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, self.middle_idx + self.num_diana_bodies, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, self.middle_idx + self.num_diana_bodies, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, self.ring_idx + self.num_diana_bodies, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, self.ring_idx + self.num_diana_bodies, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, self.little_idx + self.num_diana_bodies, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, self.little_idx + self.num_diana_bodies, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, self.thumb_idx + self.num_diana_bodies, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, self.thumb_idx + self.num_diana_bodies, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0,1,0], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.left_fingertip_state = self.rigid_body_states[:, self.left_fingertip_handles][:, :, 0:13]

        self.right_fingertip_state = self.rigid_body_states[:, self.right_fingertip_handles][:, :, 0:13]

        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()

    def compute_full_state(self, asymm_obs=False):
        num_ft_states = 65
        num_ft_states = 13 * int(self.num_fingertips / 2)  

        self.obs_buf[:, :self.num_diana_dofs] = unscale(self.left_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs: 2 * self.num_diana_dofs] = self.vel_obs_scale * self.left_diana_dof_vel
        
        fingertip_obs_start = 2 * self.num_diana_dofs
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.left_fingertip_state.reshape(self.num_envs, num_ft_states)
        
        action_obs_start = fingertip_obs_start + num_ft_states
        self.obs_buf[:, action_obs_start:action_obs_start + 13] = self.actions[:, :13]

        right_hand_start = action_obs_start + 13
        self.obs_buf[:, right_hand_start:self.num_diana_dofs + right_hand_start] = unscale(self.right_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs + right_hand_start: 2 * self.num_diana_dofs + right_hand_start] = self.vel_obs_scale * self.right_diana_dof_vel
        
        right_fingerprint_obs_start = right_hand_start + 2 * self.num_diana_dofs
        self.obs_buf[:, right_fingerprint_obs_start:right_fingerprint_obs_start + num_ft_states] = self.right_fingertip_state.reshape(self.num_envs, num_ft_states)
        
        right_action_obs_start = right_fingerprint_obs_start + num_ft_states
        self.obs_buf[:, right_action_obs_start:right_action_obs_start + 13] = self.actions[:, 13:]

        obj_obs_start = right_action_obs_start + 13
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.microwave_handle_pos
        self.obs_buf[:, obj_obs_start + 3 : obj_obs_start + 7] = self.microwave_handle_rot

    def compute_point_cloud_observation(self, collect_demonstration=False):
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
       
        self.obs_buf[:, 0:self.num_diana_dofs] = unscale(self.left_diana_dof_pos,
                                                            self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs:2*self.num_diana_dofs] = self.vel_obs_scale * self.diana_dof_vel
        
        fingertip_obs_start = 2 * self.num_diana_dofs # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
       
        hand_pose_start = fingertip_obs_start + num_ft_states
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + self.num_diana_dofs] = self.actions[:, :self.num_diana_dofs]

        right_hand_start = action_obs_start + self.num_diana_dofs
        self.obs_buf[:, right_hand_start:self.num_diana_dofs + right_hand_start] = unscale(self.diana_dof_pos,
                                                            self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs + right_hand_start:2*self.num_diana_dofs + right_hand_start] = self.vel_obs_scale * self.diana_dof_vel
        
        fingertip_obs_start = right_hand_start + 2 * self.num_diana_dofs
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        
        hand_pose_start = fingertip_obs_start + num_ft_states
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + self.num_diana_dofs] = self.actions[:, self.num_diana_dofs:]

        obj_obs_start = action_obs_start + self.num_diana_dofs  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.switch_left_handle_pos
        self.obs_buf[:, obj_obs_start + 16:obj_obs_start + 19] = self.switch_right_handle_pos
        
        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
        
        if self.camera_debug:
            import matplotlib.pyplot as plt
            self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
            camera_rgba_image = self.camera_visulization(is_depth_image=False)
            plt.imshow(camera_rgba_image)
            plt.pause(1e-9)

        for i in range(self.num_envs):
            points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, 10, self.device)
            
            if points.shape[0] > 0:
                selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
            else:
                selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
            
            point_clouds[i] = selected_points

        if self.pointCloudVisualizer != None :
            import open3d as o3d
            points = point_clouds[0, :, :3].cpu().numpy()
            self.o3d_pc.points = o3d.utility.Vector3dVector(points)
            
            if self.pointCloudVisualizerInitialized == False :
                self.pointCloudVisualizer.add_geometry(self.o3d_pc)
                self.pointCloudVisualizerInitialized = True
            else :
                self.pointCloudVisualizer.update(self.o3d_pc)

        self.gym.end_access_image_tensors(self.sim)
        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)

        point_clouds_start = obj_obs_start + 19
        self.obs_buf[:, point_clouds_start:].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))

    def reset(self, env_ids, goal_env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_diana_dofs * 2 + 1), device=self.device)
        
        # reset microwave dof
        microwave_pos = self.microwave_default_dof_pos
        self.microwave_dof_pos[env_ids, :] = microwave_pos
        self.prev_targets[env_ids, 54:] = microwave_pos
        self.cur_targets[env_ids, 54:] = microwave_pos

        # reset cube pos
        self.root_state_tensor[self.cube_indices[env_ids]] = self.cube_init_state[env_ids].clone()
        self.root_state_tensor[self.cube_indices[env_ids], 0:2] = self.cube_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.cube_indices[env_ids], self.up_axis_idx] = self.cube_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.cube_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.cube_indices[env_ids], 7:13])
      
        # hand delta computation
        left_delta_max = self.diana_dof_upper_limits - self.left_diana_default_dof_pos
        left_delta_min = self.diana_dof_lower_limits - self.left_diana_default_dof_pos
        left_rand_delta = left_delta_min + (left_delta_max - left_delta_min) * rand_floats[:, 1:1+self.num_diana_dofs]
        right_delta_max = self.diana_dof_upper_limits - self.right_diana_default_dof_pos
        right_delta_min = self.diana_dof_lower_limits - self.right_diana_default_dof_pos
        right_rand_delta = right_delta_min + (right_delta_max - right_delta_min) * rand_floats[:, 1+self.num_diana_dofs:1+self.num_diana_dofs*2]
        
        # reset left_hand
        left_pos = self.left_diana_default_dof_pos + self.reset_dof_pos_noise * left_rand_delta

        self.left_diana_dof_pos[env_ids, :] = left_pos
        self.prev_targets[env_ids, :self.num_diana_dofs] = left_pos
        self.cur_targets[env_ids, :self.num_diana_dofs] = left_pos

        left_hand_indices = self.left_hand_indices[env_ids].to(torch.int32)

        # reset right_hand
        right_pos = self.right_diana_default_dof_pos + self.reset_dof_pos_noise * right_rand_delta

        self.right_diana_dof_pos[env_ids, :] = right_pos
        self.prev_targets[env_ids, self.num_diana_dofs:self.num_diana_dofs*2] = right_pos
        self.cur_targets[env_ids, self.num_diana_dofs:self.num_diana_dofs*2] = right_pos

        right_hand_indices = self.right_hand_indices[env_ids].to(torch.int32)

        # wrapping up
        hand_indices = torch.unique(torch.cat([left_hand_indices,
                                                 right_hand_indices]).to(torch.int32))
        
        all_indices = torch.unique(torch.cat([hand_indices,
                                                 self.microwave_indices[env_ids]]).to(torch.int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_indices), len(all_indices))
                                                 
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_indices), len(all_indices))  
        all_and_obj_indices = torch.unique(torch.cat([hand_indices,
                                                 self.microwave_indices[env_ids],
                                                 self.cube_indices[env_ids]]).to(torch.int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_and_obj_indices), len(all_and_obj_indices))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        
        self.cur_targets[:, [8,12,16,20,24]] = scale(self.actions[:, 3:8], self.diana_dof_lower_limits[[8,12,16,20,24]], self.diana_dof_upper_limits[[8,12,16,20,24]])
        self.cur_targets[:, [8,12,16,20,24]] = self.act_moving_average * self.cur_targets[:, [8,12,16,20,24]] + (1.0 - self.act_moving_average) * self.prev_targets[:, [8,12,16,20,24]]
        self.cur_targets[:, [9,13,17,21,25]] = self.cur_targets[:, [10,14,18,22,26]] = scale(self.actions[:, 8:13], self.diana_dof_lower_limits[[9,13,17,21,25]], self.diana_dof_upper_limits[[9,13,17,21,25]])
        self.cur_targets[:, [9,13,17,21,25]] = self.cur_targets[:, [10,14,18,22,26]] = self.act_moving_average * self.cur_targets[:, [9,13,17,21,25]] + (1.0 - self.act_moving_average) * self.prev_targets[:, [9,13,17,21,25]]
        
        self.cur_targets[:, [35,39,43,47,51]] = scale(self.actions[:, 16:21], self.diana_dof_lower_limits[[8,12,16,20,24]], self.diana_dof_upper_limits[[8,12,16,20,24]])
        self.cur_targets[:, [35,39,43,47,51]] = self.act_moving_average * self.cur_targets[:, [35,39,43,47,51]] + (1.0 - self.act_moving_average) * self.prev_targets[:, [35,39,43,47,51]]
        self.cur_targets[:, [36,40,44,48,52]] = self.cur_targets[:, [37,41,45,49,53]] = scale(self.actions[:, 21:26], self.diana_dof_lower_limits[[9,13,17,21,25]], self.diana_dof_upper_limits[[9,13,17,21,25]])
        self.cur_targets[:, [36,40,44,48,52]] = self.cur_targets[:, [37,41,45,49,53]] = self.act_moving_average * self.cur_targets[:, [36,40,44,48,52]] + (1.0 - self.act_moving_average) * self.prev_targets[:, [36,40,44,48,52]]

        stage_2_ids = [self.progress_buf >= 15]
        stage_3_ids = [self.progress_buf > 75]

        left_pos_err = self.microwave_handle_pos - self.left_hand_pos
        left_pos_err += self.actions[:,:3] * 0.1

        right_pos_err = self.cube_pos - self.right_hand_pos
        right_pos_err += self.actions[:,13:16] * 0.1
        # right_pos_err[:,2] += 0.05
        
        # a,b,c,d = self.right_hand_rot[0]
        # quat = gymapi.Quat(a,b,c,d)
        # euler = quat.to_euler_zyx()
        # print(euler)

        self.l_target_euler = to_torch([1.57, 0, 1.57], device=self.device).repeat((self.num_envs, 1)) # 0,1.57,0
        left_target_rot = quat_from_euler_xyz(self.l_target_euler[:, 0], self.l_target_euler[:, 1], self.l_target_euler[:, 2])
        left_rot_err = orientation_error(left_target_rot, self.rigid_body_states[:, 7, 3:7].clone())
        self.r_target_euler = to_torch([-1.57,3.14,1.57], device=self.device).repeat((self.num_envs, 1)) # -1.57,3.14,1.57 aligns perfectly with coordinates # -3.14,3.14,1.57 perfectly opposite
        right_target_rot = quat_from_euler_xyz(self.r_target_euler[:, 0], self.r_target_euler[:, 1], self.r_target_euler[:, 2])
        right_rot_err = orientation_error(right_target_rot, self.rigid_body_states[:, 34, 3:7].clone())

        l_dpose = torch.cat([left_pos_err, left_rot_err], -1).unsqueeze(-1)
        l_delta = control_ik(self.l_jacobian[:, 7 - 1, :, :7], self.device, l_dpose, self.num_envs)
        r_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1)
        r_delta = control_ik(self.r_jacobian[:, 7 - 1, :, :7], self.device, r_dpose, self.num_envs)
        
        self.cur_targets[:, :7] = self.left_diana_dof_pos[:, 0:7] + l_delta[:, :7]
        self.cur_targets[:, :7][stage_2_ids] = self.prev_targets[:, :7][stage_2_ids]
        self.cur_targets[:, :7][stage_3_ids] = self.left_arm_dof_trajectory_list[0][0:7]

        self.cur_targets[:, 27:34] = self.right_diana_dof_pos[:, 0:7] + r_delta[:, :7]
        self.cur_targets[:, 27:34][stage_2_ids] = self.prev_targets[:, 27:34][stage_2_ids]
        self.cur_targets[:, 27:34][stage_3_ids] = self.left_arm_dof_trajectory_list[1][0:7]
        
        self.cur_targets[:, 7:27][stage_3_ids] = self.prev_targets[:, 7:27][stage_3_ids]
        self.cur_targets[:, 34:54][stage_3_ids] = self.prev_targets[:, 34:54][stage_3_ids]

        self.cur_targets[:, :27] = tensor_clamp(self.cur_targets[:, :27], self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.cur_targets[:, 27:54] = tensor_clamp(self.cur_targets[:, 27:54], self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.prev_targets[:,:] = self.cur_targets[:,:]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        # if True:
        #     self.gym.clear_lines(self.viewer)
        #     self.gym.refresh_rigid_body_state_tensor(self.sim)

        #     for i in range(self.num_envs):
        #         self.add_debug_lines(self.envs[i], self.microwave_handle_pos[i], self.microwave_handle_rot[i])
        #         self.add_debug_lines(self.envs[i], self.left_hand_pos[i], self.left_hand_rot[i])
        #         self.add_debug_lines(self.envs[i], self.cube_pos[i], self.cube_rot[i])
        #         self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
       
    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        return camera_image

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    depth_buffer = camera_tensor.to(device)

    vinv = camera_view_matrix_inv

    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, cube_pos, cube_rot, microwave_handle_pos, microwave_handle_rot,
    right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    actions, action_penalty_scale: float
):
    left_hand_finger_dist = (torch.norm(microwave_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(microwave_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(microwave_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(microwave_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(microwave_handle_pos - left_hand_th_pos, p=2, dim=-1))
    
    left_hand_dist_rew = torch.exp(-0.1*(left_hand_finger_dist)) * 1

    right_hand_finger_dist = (torch.norm(cube_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(cube_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(cube_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(cube_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(cube_pos - right_hand_th_pos, p=2, dim=-1))
    
    right_hand_dist_rew = torch.exp(-0.1*(right_hand_finger_dist)) * 1

    open_rew = (microwave_handle_pos[:,0] + 0.7) * 10
    
    lift_rew = (cube_pos[:, 2] - 0.575) * 10
    
    proximity_rew = torch.exp(-0.1*(torch.norm(cube_pos - microwave_handle_pos, p=2, dim=-1)))
    
    action_penalty = torch.sum(actions ** 2, dim=-1)
    
    reward = left_hand_dist_rew + right_hand_dist_rew + open_rew + lift_rew + proximity_rew - action_penalty * action_penalty_scale
    
    resets = torch.where(left_hand_finger_dist >= 4, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(right_hand_finger_dist >= 4, torch.ones_like(reset_buf), reset_buf)
    
    successes = torch.where(successes == 0, torch.where(((open_rew > 2.5) & (lift_rew > 0.25)), torch.ones_like(successes), successes), successes)

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
