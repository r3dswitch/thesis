import numpy as np
import os
import torch
import random
import glob

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from tasks.hand_base.base_task import BaseTask

def iprint(*strings):
    print(strings)
    exit()

class ArnieAffordance(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg

        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index

        self.is_multi_agent = is_multi_agent

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.hand_reset_step = self.cfg["env"]["handResetStep"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)
        
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]

        self.ignore_z = (self.object_type == "pen")

        self.obs_type = self.cfg["env"]["observationType"]

        self.fingertip_names = [    "Right_Index_Phadist",
                                    "Right_Middle_Phadist",
                                    "Right_Ring_Phadist",
                                    "Right_Thumb_Phadist",
                                    "Right_Little_Phadist" ]
        
        self.num_obs_dict = {
            "full_state": 200,
            "partial_contact": 156
        }
        self.up_axis = 'z'

        num_states = 0

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 27

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1, 0.5, 1.5)
            cam_target = gymapi.Vec3(0.2, 0.2, 0.75)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.dt = self.sim_params.dt
        
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "hand"))

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.arm_hand_dof_default_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_dof_default_pos[:7] = torch.tensor([2.08,-0.72,-1.88,1.24,-0.80,-1.22,3], dtype=torch.float, device=self.device)        

        self.arm_hand_dof_default_pos[7:] = to_torch([0, 0, 0, 0, 
                                                      0, 0, 0, 0, 
                                                      0, 0, 0, 0, 
                                                      0, 0, 0, 0, 
                                                      0, 0, 0, 0], dtype=torch.float, device=self.device)

        self.arm_dof_trajectory_list = []
        self.arm_dof_trajectory = to_torch([1.93,-0.77,-2.00,1.58,-1.21,-1.52,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float, device=self.device)
        self.arm_dof_trajectory_list.append(self.arm_dof_trajectory)
        self.arm_dof_trajectory = to_torch([0.24,-1.11,-2.27,1.24,-1.95,-0.95,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float, device=self.device)
        self.arm_dof_trajectory_list.append(self.arm_dof_trajectory)

        self.target_euler = to_torch([0,0,0], device=self.device).repeat((self.num_envs, 1)) # 0,0,0 perpendicular, 0,-1.57,0 horizontal
        
        self.dataset_basedir = "/home/smondal/Desktop/DexterousHands/affordpose_data/Dataset/Single/validated"
        self.mesh_folder = "/home/smondal/Desktop/DexterousHands/affordpose_data/Dataset/Single/manifold"
        self.pt_dataset_path = glob.glob(os.path.join(self.dataset_basedir, "cmap_dataset_*.pt"))
        
        self.grasp_list = []
        for dataset_path_i in self.pt_dataset_path:
            graspdata = torch.load(dataset_path_i)["metadata"]
            for index_candidate in range(self.num_envs):
                hand_pose = graspdata[index_candidate][1]           # hand pose
                self.grasp_list.append(hand_pose)

        self.grasp_list = torch.stack(self.grasp_list).to(self.device)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.success1 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.success2 = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.wrist_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "link_7", gymapi.DOMAIN_ENV)
        
        self.cube_pos = self.root_state_tensor[self.cube_indices, 0:3].clone()
        self.cube_rot = self.root_state_tensor[self.cube_indices, 3:7].clone()
        self.wrist_pos = self.root_state_tensor[self.hand_indices, 0:3].clone()
        self.wrist_rot = self.root_state_tensor[self.hand_indices, 3:7].clone()
        
        self.rot = torch.tensor([0.707,0.707,0,0],device=self.device).repeat(self.num_envs, 1)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        self.sim_params.gravity.z = -9.8 

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')

        arm_hand_asset_file = "urdf/right_hit.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01
        
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)
        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        
        self.actuated_dof_indices = [i for i in range(7, self.num_arm_hand_dofs)]

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        for i in range(27):
            robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if i < 7:
                robot_dof_props['stiffness'][i] = 400
                robot_dof_props['effort'][i] = 200
                robot_dof_props['damping'][i] = 80
            else:
                robot_dof_props['velocity'][i] = 10.0
                robot_dof_props['effort'][i] = 0.7
                robot_dof_props['stiffness'][i] = 20
                robot_dof_props['damping'][i] = 1

            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_lower_qvel = to_torch(-robot_dof_props["velocity"], device=self.device)
        self.arm_hand_dof_upper_qvel = to_torch(robot_dof_props["velocity"], device=self.device)

        for i in range(self.num_arm_hand_dofs):
            self.arm_hand_dof_default_vel.append(0.0)

        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.35, 0.0, 0.6)
        arm_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        table_dims = gymapi.Vec3(1.5, 1.5, 0.6)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.2, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)
        
        cube_asset_root = "/home/smondal/Desktop/DexterousHands/affordpose_data/Dataset/Single/urdf"
        cube_file_name = "AffordPose_bottle_3520_1000.urdf"

        self.num_object_bodies = 0
        self.num_object_shapes = 0
        
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.vhacd_enabled = True
        cube_asset_options.fix_base_link = True
        cube_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        cube_asset_options.vhacd_params = gymapi.VhacdParams()
        cube_asset_options.vhacd_params.resolution = 500000
        cube_asset_options.thickness = 0.0001
        cube_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        cube_asset = self.gym.load_asset(self.sim, cube_asset_root, cube_file_name, cube_asset_options)

        self.num_object_bodies += self.gym.get_asset_rigid_body_count(cube_asset)
        self.num_object_shapes += self.gym.get_asset_rigid_shape_count(cube_asset)
        
        cube_start_pose = gymapi.Transform()
        cube_start_pose.p = gymapi.Vec3(0.3,0.2,1.25) # x=0.4 control variable, x=0.3 good for perpendicular
        cube_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)
               
        max_agg_bodies = self.num_arm_hand_bodies + 100
        max_agg_shapes = self.num_arm_hand_shapes + 100
        
        self.envs = []

        self.cube_init_states = []
        self.hand_init_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.table_indices = []
        self.cube_indices = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, -1, 0)
            self.hand_init_states.append([arm_hand_start_pose.p.x,
                                           arm_hand_start_pose.p.y,
                                           arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x,
                                           arm_hand_start_pose.r.y,
                                           arm_hand_start_pose.r.z,
                                           arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            arm_hand_actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, arm_hand_actor)
            for arm_hand_actor_shape_prop in arm_hand_actor_shape_props:
                arm_hand_actor_shape_prop.friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, arm_hand_actor, arm_hand_actor_shape_props)

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)
            
            cube_handle = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 0, 1)
            self.cube_init_states.append([cube_start_pose.p.x, cube_start_pose.p.y, cube_start_pose.p.z,
                                            cube_start_pose.r.x, cube_start_pose.r.y, cube_start_pose.r.z, cube_start_pose.r.w,
                                            0, 0, 0, 0, 0, 0])
            cube_idx = self.gym.get_actor_index(env_ptr, cube_handle, gymapi.DOMAIN_SIM)

            self.cube_indices.append(cube_idx)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.fingertip_handles = [self.gym.find_actor_rigid_body_handle(env_ptr, arm_hand_actor, name) for name in self.fingertip_names]
        
        self.hand_init_states = to_torch(self.hand_init_states, device=self.device).view(self.num_envs, 13)
        self.cube_init_states = to_torch(self.cube_init_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.cube_indices = to_torch(self.cube_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.success1[:], self.success2[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.success1, self.success2, self.consecutive_successes, 
            self.max_episode_length, self.cube_pos, self.wrist_pos, self.arm_hand_ff_pos, self.arm_hand_rf_pos, self.arm_hand_mf_pos, self.arm_hand_th_pos, self.arm_hand_lf_pos,
            self.actions, self.max_consecutive_successes, self.av_factor
        )

        self.extras['Grasp Success'] = self.success1
        self.extras['Lift Success'] = self.success2
        self.extras['Consecutive Successes'] = self.consecutive_successes

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.wrist_pose = self.rigid_body_states[:, self.wrist_rigid_body_index, 0:7]
        self.wrist_pos = self.rigid_body_states[:, self.wrist_rigid_body_index, 0:3]
        self.wrist_rot = self.rigid_body_states[:, self.wrist_rigid_body_index, 3:7]
        self.wrist_rot = quat_mul(self.rigid_body_states[:, self.wrist_rigid_body_index, 3:7], self.rot)
        self.wrist_linvel = self.rigid_body_states[:, self.wrist_rigid_body_index, 7:10]
        self.wrist_angvel = self.rigid_body_states[:, self.wrist_rigid_body_index, 10:13]

        self.cube_pose = self.root_state_tensor[self.cube_indices, 0:7]
        self.cube_pos = self.root_state_tensor[self.cube_indices, 0:3]
        self.cube_rot = self.root_state_tensor[self.cube_indices, 3:7]
        self.cube_linvel = self.root_state_tensor[self.cube_indices, 7:10]
        self.cube_angvel = self.root_state_tensor[self.cube_indices, 10:13]
        
        self.arm_hand_ff_pos = self.rigid_body_states[:, 11, 0:3]
        self.arm_hand_ff_rot = self.rigid_body_states[:, 11, 3:7]
        self.arm_hand_ff_linvel = self.rigid_body_states[:, 11, 7:10]
        self.arm_hand_ff_angvel = self.rigid_body_states[:, 11, 10:13]

        self.arm_hand_mf_pos = self.rigid_body_states[:, 19, 0:3]
        self.arm_hand_mf_rot = self.rigid_body_states[:, 19, 3:7]
        self.arm_hand_mf_linvel = self.rigid_body_states[:, 19, 7:10]
        self.arm_hand_mf_angvel = self.rigid_body_states[:, 19, 10:13]

        self.arm_hand_rf_pos = self.rigid_body_states[:, 23, 0:3]
        self.arm_hand_rf_rot = self.rigid_body_states[:, 23, 3:7]
        self.arm_hand_rf_linvel = self.rigid_body_states[:, 23, 7:10]
        self.arm_hand_rf_angvel = self.rigid_body_states[:, 23, 10:13]
        
        self.arm_hand_th_pos = self.rigid_body_states[:, 27, 0:3]
        self.arm_hand_th_rot = self.rigid_body_states[:, 27, 3:7]
        self.arm_hand_th_linvel = self.rigid_body_states[:, 27, 7:10]
        self.arm_hand_th_angvel = self.rigid_body_states[:, 27, 10:13]

        self.arm_hand_lf_pos = self.rigid_body_states[:, 15, 0:3]
        self.arm_hand_lf_rot = self.rigid_body_states[:, 15, 3:7]
        self.arm_hand_lf_linvel = self.rigid_body_states[:, 15, 7:10]
        self.arm_hand_lf_angvel = self.rigid_body_states[:, 15, 10:13]

        self.arm_hand_ff_state = self.rigid_body_states[:, 11, 0:13]
        self.arm_hand_mf_state = self.rigid_body_states[:, 19, 0:13]
        self.arm_hand_rf_state = self.rigid_body_states[:, 23, 0:13]
        self.arm_hand_th_state = self.rigid_body_states[:, 27, 0:13]
        self.arm_hand_lf_state = self.rigid_body_states[:, 15, 0:13]

        self.arm_hand_ff_pos = self.arm_hand_ff_pos + quat_apply(self.arm_hand_ff_rot[:], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)
        self.arm_hand_mf_pos = self.arm_hand_mf_pos + quat_apply(self.arm_hand_mf_rot[:], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)
        self.arm_hand_rf_pos = self.arm_hand_rf_pos + quat_apply(self.arm_hand_rf_rot[:], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)
        self.arm_hand_th_pos = self.arm_hand_th_pos + quat_apply(self.arm_hand_th_rot[:], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)
        self.arm_hand_lf_pos = self.arm_hand_lf_pos + quat_apply(self.arm_hand_lf_rot[:], to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.03)

    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0,1.0, (len(env_ids),self.num_arm_hand_dofs*2), device=self.device)
        # reset cube
        self.root_state_tensor[self.cube_indices[env_ids]] = self.cube_init_states[env_ids].clone()
        self.root_state_tensor[self.cube_indices[env_ids], 0:2] = self.cube_init_states[env_ids, 0:2] 
        self.root_state_tensor[self.cube_indices[env_ids], self.up_axis_idx] = self.cube_init_states[env_ids, self.up_axis_idx] 

        self.root_state_tensor[self.cube_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.cube_indices[env_ids], 7:13])

        cube_indices = torch.unique(self.cube_indices[env_ids]).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(cube_indices), len(cube_indices))
        
        # reset arm
        delta_max = self.arm_hand_dof_upper_limits - self.arm_hand_dof_default_pos
        delta_min = self.arm_hand_dof_lower_limits - self.arm_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_arm_hand_dofs]

        pos = self.arm_hand_dof_default_pos #+ self.reset_dof_pos_noise * rand_delta

        self.arm_hand_dof_pos[env_ids, :] = pos
        
        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = pos #self.arm_hand_dof_pos[env_ids]
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = pos #self.arm_hand_dof_pos[env_ids]

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.prev_targets),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        all_indices = torch.unique(torch.cat([hand_indices,
                                              cube_indices]).to(torch.int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success1[env_ids] = 0
        self.success2[env_ids] = 0

    def pre_physics_step(self, actions):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        
        pos_err = self.cube_pos + self.grasp_list[:, 0:3] - self.wrist_pos
            
        target_rot = to_torch(sixd_to_quaternion(self.grasp_list[:, 3:9].cpu()), device=self.device)
        target_rot = quat_mul(target_rot, self.cube_rot)
        rot_err = orientation_error(target_rot, self.wrist_rot)
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_tensor[:, 7 - 1, :, :7], self.device, dpose, self.num_envs)
        
        self.cur_targets[:, :7] = self.arm_hand_dof_pos[:, 0:7] + delta[:, :7]
        self.cur_targets[:, 7:27] = self.grasp_list[:, 8:28]
        
        self.cur_targets[:, :27] = tensor_clamp(self.cur_targets[:, :27], self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
        
        self.prev_targets = self.cur_targets

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if True:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                hand_rot = to_torch(sixd_to_quaternion(self.grasp_list[:, 3:9].cpu()), device=self.device)
                self.add_debug_lines(self.envs[i], 
                                     self.cube_pos[i] + self.grasp_list[i, 0:3], 
                                     quat_mul(hand_rot[i], self.cube_rot[i]))
                self.add_debug_lines(self.envs[i], self.wrist_pos[i], self.wrist_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def print_euler(self, rot):
        a,b,c,d = rot[0]
        quat = gymapi.Quat(a,b,c,d)
        euler = quat.to_euler_zyx()
        print(euler)
    
@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, success1, success2, consecutive_successes,
    max_episode_length: float, cube_pos, wrist_pos, arm_hand_ff_pos, arm_hand_rf_pos, arm_hand_mf_pos, arm_hand_th_pos, arm_hand_lf_pos,
    actions, max_consecutive_successes: int, av_factor: float
):
    arm_hand_finger_dist = (torch.norm(cube_pos - arm_hand_ff_pos, p=2, dim=-1) + torch.norm(cube_pos - arm_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(cube_pos - arm_hand_rf_pos, p=2, dim=-1) + torch.norm(cube_pos - arm_hand_th_pos, p=2, dim=-1) + torch.norm(cube_pos - arm_hand_lf_pos, p=2, dim=-1))

    dist_reward = torch.exp(- 1 * (arm_hand_finger_dist))
    
    
    cube_hold_rew = (cube_pos[:, 2] - 0.625) * 10
    
    resets = torch.where(arm_hand_finger_dist >= 5, torch.ones_like(reset_buf), reset_buf)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    reward = cube_hold_rew + dist_reward
    
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum((success1 * resets.float() + success2 * resets.float())/2)
    
    success1 = torch.where(success1 == 0, torch.where((dist_reward > 0.5), torch.ones_like(success1), success1), success1)
    success2 = torch.where(success2 == 0, torch.where((cube_hold_rew > 0.1), torch.ones_like(success2), success2), success2)
    
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, reset_goal_buf, progress_buf, success1, success2, cons_successes

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))

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

def sixd_to_quaternion(tensor_of_sixd_vectors):
    from scipy.spatial.transform import Rotation as R
    Q = []
    for vector in tensor_of_sixd_vectors:
        # Step 1: Separate the 6D vector into two 3D vectors
        r1 = vector[:3]
        r2 = vector[3:]

        # Step 2: Normalize the first vector to ensure it is a unit vector
        r1 = r1 / np.linalg.norm(r1)

        # Step 3: Make the second vector orthogonal to the first
        r2 = r2 - np.dot(r1, r2) * r1
        r2 = r2 / np.linalg.norm(r2)

        # Step 4: Compute the third vector using cross product to ensure orthogonality
        r3 = np.cross(r1, r2)

        # Step 5: Construct the rotation matrix using the three orthogonal vectors
        rotation_matrix = np.stack([r1, r2, r3], axis=-1)

        # Step 6: Convert the rotation matrix to a quaternion
        rotation = R.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # returns [x, y, z, w]
        Q.append(quaternion)
    Q = [torch.from_numpy(arr) for arr in Q]
    Q = torch.stack(Q)
    return Q