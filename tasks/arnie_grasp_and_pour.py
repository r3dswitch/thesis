from matplotlib.pyplot import axis
import numpy as np
from PIL import Image as Im
 
import os
import random
import torch

from bidexhands.utils.torch_jit_utils import *
from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

def iprint(*strings):
    print(strings)
    exit()

class ArnieGraspAndPour(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg # from cfg/filename.yaml
        self.sim_params = sim_params
        self.physics_engine = physics_engine # physx
        
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"] # 50
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"] # 1.0
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"] # -0.0002
        self.success_tolerance = self.cfg["env"]["successTolerance"] # 0.1
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"] # 250
        self.fall_dist = self.cfg["env"]["fallDistance"] # 0.4
        self.fall_penalty = self.cfg["env"]["fallPenalty"] # 0
        self.rot_eps = self.cfg["env"]["rotEps"] # 0.1

        self.vel_obs_scale = 0.2  

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"] # 0.01
        self.reset_dof_pos_noise = 0.01 # self.cfg["env"]["resetDofPosRandomInterval"] # 0.05

        self.debug_viz = self.cfg["env"]["enableDebugVis"] # False
 
        self.max_episode_length = self.cfg["env"]["episodeLength"] # 75
        self.reset_time = self.cfg["env"].get("resetTime", -1.0) # -1.0
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"] # 0
        self.av_factor = self.cfg["env"].get("averFactor", 0.01) # 0.01

        self.ignore_z = False

        self.obs_type = self.cfg["env"]["observationType"] # full_state

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": 398 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 398 + self.num_point_cloud_feature_dim * 3,
            "full_state": 316 # num_hand_obs * 2
        }
        
        self.num_hand_obs = 158 # 54 + 65 + 27
        self.up_axis = 'z'

        self.left_fingertips = self.right_fingertips = ["Right_Index_Phadist", "Right_Middle_Phadist", "Right_Ring_Phadist", "Right_Little_Phadist", "Right_Thumb_Phadist"]

        self.num_fingertips = 10

        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"] # False

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

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

        if self.obs_type in ["point_cloud"]:
            from PIL import Image as Im
            from bidexhands.utils import o3dviewer

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim) # 4,13
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim) # 54,2
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim) # 61,13

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.left_diana_default_dof_pos = to_torch([-1.1, # -0.9
                -1.4,
                -3.1,
                0.8,
                3.1,
                0.5,
                -0.4,
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
        self.right_diana_default_dof_pos = to_torch([-1.7,
                1.0,
                0,
                1.0,
                3.1,
                0.5,
                1.5,
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
        
        self.left_diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_diana_dofs]
        self.left_diana_dof_pos = self.left_diana_dof_state[..., 0]
        self.left_diana_dof_vel = self.left_diana_dof_state[..., 1]

        self.right_diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_diana_dofs:self.num_diana_dofs*2]
        self.right_diana_dof_pos = self.right_diana_dof_state[..., 0]
        self.right_diana_dof_vel = self.right_diana_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
       
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

        left_diana_asset_file = "urdf/right_hit.urdf"
        right_diana_asset_file = "urdf/right_hit.urdf"
        object_asset_file = "urdf/cup/cup.urdf"
        table_asset_file = "urdf/square_table.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        left_diana_asset = self.gym.load_asset(self.sim, asset_root, left_diana_asset_file, asset_options)
        right_diana_asset = self.gym.load_asset(self.sim, asset_root, right_diana_asset_file, asset_options)

        self.num_diana_bodies = self.gym.get_asset_rigid_body_count(left_diana_asset) # 28
        self.num_diana_shapes = self.gym.get_asset_rigid_shape_count(left_diana_asset) # 29
        self.num_diana_dofs = self.gym.get_asset_dof_count(left_diana_asset) # 27
        
        self.actuated_dof_indices = to_torch([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26])

        left_diana_dof_props = self.gym.get_asset_dof_properties(left_diana_asset)
        right_diana_dof_props = self.gym.get_asset_dof_properties(right_diana_asset)

        self.diana_dof_lower_limits = []
        self.diana_dof_upper_limits = []
        self.diana_dof_default_pos = []
        self.diana_dof_default_vel = []

        for i in range(self.num_diana_dofs):
            self.diana_dof_lower_limits.append(left_diana_dof_props['lower'][i])
            self.diana_dof_upper_limits.append(left_diana_dof_props['upper'][i])
            self.diana_dof_default_pos.append(0.0)
            self.diana_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.diana_dof_lower_limits = to_torch(self.diana_dof_lower_limits, device=self.device)
        self.diana_dof_upper_limits = to_torch(self.diana_dof_upper_limits, device=self.device)
        self.diana_dof_default_pos = to_torch(self.diana_dof_default_pos, device=self.device)
        self.diana_dof_default_vel = to_torch(self.diana_dof_default_vel, device=self.device)

        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.fix_base_link= True
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        table_asset = self.gym.load_asset(self.sim, asset_root, table_asset_file, asset_options)

        left_diana_start_pose = gymapi.Transform()
        left_diana_start_pose.p = gymapi.Vec3(0, -0.1, 1.45)
        left_diana_start_pose.r = gymapi.Quat().from_euler_zyx(1.5652925671162337, 0, 0.227)

        right_diana_start_pose = gymapi.Transform()
        right_diana_start_pose.p = gymapi.Vec3(0, 0.1, 1.45)
        right_diana_start_pose.r = gymapi.Quat().from_euler_zyx(-1.5652925671162337, 0, -0.227)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(-0.4,0,0.7)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(1.5652925671162337, 0, 0)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(-0.5,0,0.2)

        max_agg_bodies = self.num_diana_bodies * 2 + 5
        max_agg_shapes = self.num_diana_shapes * 2 + 5

        self.dianas = []
        self.envs = []

        self.object_init_state = []
        self.table_init_state = []
        self.hand_init_state = []

        self.left_hand_indices = []
        self.right_hand_indices = []
        
        self.fingertip_indices = []
        self.object_indices = []
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
            right_diana_actor = self.gym.create_actor(env_ptr, right_diana_asset, right_diana_start_pose, "right_hand", i, -1, 0)
            
            self.hand_init_state.append([left_diana_start_pose.p.x, left_diana_start_pose.p.y, left_diana_start_pose.p.z,
                                           left_diana_start_pose.r.x, left_diana_start_pose.r.y, left_diana_start_pose.r.z, left_diana_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, left_diana_actor, left_diana_dof_props)
            left_hand_idx = self.gym.get_actor_index(env_ptr, left_diana_actor, gymapi.DOMAIN_SIM)
            self.left_hand_indices.append(left_hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, right_diana_actor, right_diana_dof_props)
            right_hand_idx = self.gym.get_actor_index(env_ptr, right_diana_actor, gymapi.DOMAIN_SIM)
            self.right_hand_indices.append(right_hand_idx)            
            
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 0, 0)
            self.table_init_state.append([table_start_pose.p.x, table_start_pose.p.y, table_start_pose.p.z,
                                           table_start_pose.r.x, table_start_pose.r.y, table_start_pose.r.z, table_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            if self.obs_type in ["point_cloud"]:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.25, -0.5, 0.75), gymapi.Vec3(-0.24, -0.5, 0))
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

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.hand_init_state = to_torch(self.hand_init_state, device=self.device).view(self.num_envs, 13)

        self.left_fingertip_handles = to_torch(self.left_fingertip_handles, dtype=torch.long, device=self.device)
        self.right_fingertip_handles = to_torch(self.right_fingertip_handles, dtype=torch.long, device=self.device)
        
        self.left_hand_indices = to_torch(self.left_hand_indices, dtype=torch.long, device=self.device)
        self.right_hand_indices = to_torch(self.right_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        
        # IK Test
        self.object_pos = self.object_init_state[:,:3]
        self.object_rot = self.object_init_state[:,3:7]
        
        self.left_hand_pos = self.hand_init_state[:,:3]
        self.left_hand_rot = self.hand_init_state[:,3:7]

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "left_hand")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        self.hand_index = self.gym.get_asset_rigid_body_dict(left_diana_asset)["link_7"]
        self.j_eef = self.jacobian[:, self.hand_index - 1, :, :7]
        self.init_data()
        # IK Test Ends
    
    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.dianas[0], "link_7")
        
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        
        bot_hand_pos = to_torch([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        bot_hand_rot = to_torch([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.left_hand_pos = torch.zeros_like(bot_hand_pos)
        self.left_hand_rot = torch.zeros_like(bot_hand_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.left_hand_pos, self.left_hand_rot, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, False
        )

    def compute_observations(self):
        self.gym.simulate(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.left_fingertip_state = self.rigid_body_states[:, self.left_fingertip_handles][:, :, 0:13]
        self.left_fingertip_pos = self.rigid_body_states[:, self.left_fingertip_handles][:, :, 0:3]
        
        self.right_fingertip_state = self.rigid_body_states[:, self.right_fingertip_handles][:, :, 0:13]
        self.right_fingertip_pos = self.rigid_body_states[:, self.right_fingertip_handles][:, :, 0:3]
        
        # IK Test
        link_idx =  7
        self.left_hand_pos = self.rigid_body_states[:, link_idx, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, link_idx, 3:7]
        # IK Test Ends

        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()
        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        num_ft_states = 13 * int(self.num_fingertips / 2)  

        self.obs_buf[:, :self.num_diana_dofs] = unscale(self.left_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs: 2 * self.num_diana_dofs] = self.vel_obs_scale * self.left_diana_dof_vel
        
        fingertip_obs_start = 2 * self.num_diana_dofs
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.left_fingertip_state.reshape(self.num_envs, num_ft_states)
        
        action_obs_start = fingertip_obs_start + num_ft_states
        self.obs_buf[:, action_obs_start:action_obs_start + self.num_diana_dofs] = self.actions[:, :self.num_diana_dofs]

        right_hand_start = action_obs_start + self.num_diana_dofs
        self.obs_buf[:, right_hand_start:self.num_diana_dofs + right_hand_start] = unscale(self.right_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs + right_hand_start: 2 * self.num_diana_dofs + right_hand_start] = self.vel_obs_scale * self.right_diana_dof_vel
        
        right_fingerprint_obs_start = right_hand_start + 2 * self.num_diana_dofs
        self.obs_buf[:, right_fingerprint_obs_start:right_fingerprint_obs_start + num_ft_states] = self.right_fingertip_state.reshape(self.num_envs, num_ft_states)
        
        right_action_obs_start = right_fingerprint_obs_start + num_ft_states
        # self.obs_buf[:, right_action_obs_start:right_action_obs_start + self.num_diana_dofs] = self.actions[:, self.num_diana_dofs:]

        obj_obs_start = right_action_obs_start + self.num_diana_dofs
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

    def compute_point_cloud_observation(self, collect_demonstration=False):
        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65

        self.obs_buf[:, :self.num_diana_dofs] = unscale(self.left_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs : 2 * self.num_diana_dofs] = self.vel_obs_scale * self.left_diana_dof_vel

        fingertip_obs_start = 2 * self.num_diana_dofs  
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.left_fingertip_state.reshape(self.num_envs, num_ft_states)

        action_obs_start = fingertip_obs_start + + num_ft_states
        self.obs_buf[:, action_obs_start:action_obs_start + self.num_diana_dofs] = self.actions[:, :self.num_diana_dofs]

        right_hand_start = action_obs_start + self.num_diana_dofs
        self.obs_buf[:, right_hand_start:self.num_diana_dofs + right_hand_start] = unscale(self.right_diana_dof_pos, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.obs_buf[:, self.num_diana_dofs + right_hand_start : 2 * self.num_diana_dofs + right_hand_start] = self.vel_obs_scale * self.right_diana_dof_vel
        
        right_fingerprint_obs_start = right_hand_start + 2 * self.num_diana_dofs
        self.obs_buf[:, right_fingerprint_obs_start:right_fingerprint_obs_start + num_ft_states] = self.right_fingertip_state.reshape(self.num_envs, num_ft_states)
        

        right_action_obs_start = right_fingerprint_obs_start + num_ft_states
        self.obs_buf[:, right_action_obs_start:right_action_obs_start + self.num_diana_dofs] = self.actions[:, self.num_diana_dofs:]

        obj_obs_start = right_action_obs_start + self.num_diana_dofs  
        if collect_demonstration:
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        goal_obs_start = obj_obs_start + 13  
        self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

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

        point_clouds_start = goal_obs_start + 11
        self.obs_buf[:, point_clouds_start:].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))

    def reset(self, env_ids):
        print("resets")
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_diana_dofs * 2), device=self.device)

        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        # new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        # self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)

        delta_max = self.diana_dof_upper_limits - self.diana_dof_default_pos
        delta_min = self.diana_dof_lower_limits - self.diana_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, :self.num_diana_dofs]

        pos_left = self.left_diana_default_dof_pos  + self.reset_dof_pos_noise * rand_delta
        pos_right = self.right_diana_default_dof_pos  + self.reset_dof_pos_noise * rand_delta

        self.left_diana_dof_pos[env_ids, :] = pos_left
        self.right_diana_dof_pos[env_ids, :] = pos_right

        self.prev_targets[env_ids, :self.num_diana_dofs] = pos_left
        self.cur_targets[env_ids, :self.num_diana_dofs] = pos_left

        self.prev_targets[env_ids, self.num_diana_dofs:self.num_diana_dofs * 2] = pos_right
        self.cur_targets[env_ids, self.num_diana_dofs:self.num_diana_dofs * 2] = pos_right

        left_hand_indices = self.left_hand_indices[env_ids].to(torch.int32)
        right_hand_indices = self.right_hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([left_hand_indices, right_hand_indices]).to(torch.int32))
        all_indices = torch.unique(torch.cat([all_hand_indices, object_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def reset_alt(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset(env_ids)

        self.gym.refresh_jacobian_tensors(self.sim)

        self.actions = actions.clone().to(self.device)
        
        # """
        self.target_pos = self.object_pos.clone()
        self.target_rot = self.object_rot.clone()
        
        pos_err = self.target_pos - self.left_hand_pos
        orn_err = orientation_error(self.target_rot, self.left_hand_rot)

        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  
        j_eef_T = torch.transpose(self.j_eef, 1, 2)
        damping = 0.05
        lmbda = torch.eye(6, device=self.device) * (damping ** 2)
        u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)     
        
        targets = self.prev_targets[:, :self.num_diana_dofs]
        
        targets[:, :7] = self.left_diana_dof_pos[:,:7] + u.squeeze(-1)
        # input(targets[0])
        self.cur_targets[:, :self.num_diana_dofs] = tensor_clamp(targets, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        self.prev_targets = self.cur_targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        # """
        
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

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

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    depth_buffer = camera_tensor.to(device)

    vinv = camera_view_matrix_inv

    proj = camera_proj_matrix
    fu = 2 / proj[0, 0]
    fv = 2 / proj[1, 1]

    centerU = width / 2
    centerV = height / 2

    Z = depth_buffer
    X = -(u-centerU) / width * Z * fu
    Y = (v-centerV) / height * Z * fv

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
    rew_buf, reset_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, hand_pos, hand_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    goal_dist = torch.norm(hand_pos - object_pos, p=2, dim=-1)
    
    dist_rew = goal_dist

    reward = torch.exp(-0.2 * (dist_rew * dist_reward_scale))

    successes = torch.where(successes == 0, torch.where(goal_dist < 0.1, torch.ones_like(successes), successes), successes)

    resets = torch.where(goal_dist >= 5, torch.ones_like(reset_buf), reset_buf)
   
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes).mean()

    return reward, resets, progress_buf, successes, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)