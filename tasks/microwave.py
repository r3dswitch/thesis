from bidexhands.tasks.hand_base.base_task import BaseTask

from .Math_Helpers import *
from .Isaac_Creator import *
from .Init_Creator import *

class CustomConfigFactory(ConfigFactory):
    @staticmethod
    def initialize_config(cfg, device_type, device_id, headless):
        """Initialize configuration parameters."""
        my_cfg = cfg
        my_cfg["env"]["numActions"] = 26
        my_cfg["env"]["numObservations"] = 148
        my_cfg["device_type"] = device_type
        my_cfg["device_id"] = device_id
        my_cfg["headless"] = headless
        return my_cfg
    
    @staticmethod
    def setup_tensors(obj):
        obj.actor_root_state_tensor = obj.gym.acquire_actor_root_state_tensor(obj.sim)
        obj.dof_state_tensor = obj.gym.acquire_dof_state_tensor(obj.sim)
        obj.rigid_body_tensor = obj.gym.acquire_rigid_body_state_tensor(obj.sim)
        obj.left_jacobian = gymtorch.wrap_tensor(obj.gym.acquire_jacobian_tensor(obj.sim, obj.cfg["env"]["leftJacobianActorName"]))
        obj.right_jacobian = gymtorch.wrap_tensor(obj.gym.acquire_jacobian_tensor(obj.sim, obj.cfg["env"]["rightJacobianActorName"]))

        obj.gym.refresh_actor_root_state_tensor(obj.sim)
        obj.gym.refresh_dof_state_tensor(obj.sim)
        obj.gym.refresh_rigid_body_state_tensor(obj.sim)
        
        obj.dof_state = gymtorch.wrap_tensor(obj.dof_state_tensor)
        obj.left_hand_dof_state = obj.dof_state.view(obj.num_envs, -1, 2)[:, :27]
        obj.left_hand_dof_pos = obj.left_hand_dof_state[..., 0]
        obj.left_hand_dof_vel = obj.left_hand_dof_state[..., 1]

        obj.right_hand_dof_state = obj.dof_state.view(obj.num_envs, -1, 2)[:, 27:54]
        obj.right_hand_dof_pos = obj.right_hand_dof_state[..., 0]
        obj.right_hand_dof_vel = obj.right_hand_dof_state[..., 1]
        
        obj.microwave_dof_state = obj.dof_state.view(obj.num_envs, -1, 2)[:, 54:]
        obj.microwave_dof_pos = obj.microwave_dof_state[..., 0]
        obj.microwave_dof_vel = obj.microwave_dof_state[..., 1]
        
        obj.rigid_body_states = gymtorch.wrap_tensor(obj.rigid_body_tensor).view(obj.num_envs, -1, 13)
        obj.num_bodies = obj.rigid_body_states.shape[1]
        
        obj.root_state_tensor = gymtorch.wrap_tensor(obj.actor_root_state_tensor).view(-1, 13)
        
        obj.num_dofs = obj.gym.get_sim_dof_count(obj.sim) // obj.num_envs
        obj.prev_targets = torch.zeros((obj.num_envs, obj.num_dofs), dtype=torch.float, device=obj.device)
        obj.cur_targets = torch.zeros((obj.num_envs, obj.num_dofs), dtype=torch.float, device=obj.device)
        
        obj.global_indices = torch.arange(obj.num_envs * 3, dtype=torch.int32, device=obj.device).view(obj.num_envs, -1)
        obj.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=obj.device).repeat((obj.num_envs, 1))
        obj.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=obj.device).repeat((obj.num_envs, 1))
        obj.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=obj.device).repeat((obj.num_envs, 1))
        
        obj.microwave_pos = obj.root_state_tensor[obj.microwave_indices, 0:3].clone()
        obj.microwave_rot = obj.microwave_init_rot = obj.root_state_tensor[obj.microwave_indices, 3:7].clone()
        obj.microwave_init_pos = obj.grasp_pos = obj.intermediate_pos = obj.handle_pos = obj.microwave_pos
        obj.microwave_init_rot = obj.grasp_rot = obj.intermediate_rot = obj.handle_rot = obj.microwave_rot
        
        obj.reset_goal_buf = obj.reset_buf.clone()
        obj.left_grasp_success = obj.right_grasp_success = obj.left_open_success = obj.right_lift_success = obj.right_reach_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.consecutive_successes = torch.zeros(1, dtype=torch.float, device=obj.device)
        obj.microwave_handle_pos = obj.root_state_tensor[obj.microwave_indices, 0:3].clone()
        obj.microwave_handle_rot = obj.root_state_tensor[obj.microwave_indices, 3:7].clone()
        obj.left_hand_pos = obj.root_state_tensor[obj.left_hand_indices, 0:3].clone()
        obj.left_hand_rot = obj.root_state_tensor[obj.left_hand_indices, 3:7].clone()
        obj.right_hand_pos = obj.root_state_tensor[obj.right_hand_indices, 0:3].clone()
        obj.right_hand_rot = obj.root_state_tensor[obj.right_hand_indices, 3:7].clone()
        obj.cube_pos = obj.root_state_tensor[obj.cube_indices, 0:3].clone()
        obj.cube_rot = obj.root_state_tensor[obj.cube_indices, 3:7].clone()
        
    @staticmethod
    def setup_hand_positions(obj):
        left_init_pos = obj.cfg["env"]["handPositions"]["leftDefaultPos"]
        right_init_pos = obj.cfg["env"]["handPositions"]["rightDefaultPos"]
        
        left_trajectory = obj.cfg["env"]["handPositions"]["leftTrajectory"]
        right_trajectory = obj.cfg["env"]["handPositions"]["rightTrajectory"]
        
        obj.left_hand_default_pos = obj.right_hand_default_pos = torch.zeros(
            obj.num_hand_dofs, dtype=torch.float, device=obj.device
        )
        
        obj.left_hand_default_pos[:7] = torch.tensor( left_init_pos, dtype=torch.float, device=obj.device )
        obj.right_hand_default_pos[:7] = torch.tensor( right_init_pos, dtype=torch.float, device=obj.device )
        
        obj.left_hand_default_pos[7:27] = obj.right_hand_default_pos[7:27] = torch.tensor( [0,0.6,0.6,0,0,0.6,0.6,0,0,0.6,0.6,0,0,0.6,0.6,0,0,0.6,0.6,0], dtype=torch.float, device=obj.device )
        
        obj.left_trajectory = [to_torch(pos, dtype=torch.float, device=obj.device) for pos in left_trajectory]
        obj.right_trajectory = [to_torch(pos, dtype=torch.float, device=obj.device) for pos in right_trajectory]
    
    @staticmethod
    def setup_microwave_position(obj):
        obj.microwave_default_dof_pos = torch.zeros(
            obj.num_microwave_dofs, dtype=torch.float, device=obj.device
        )
        
class MicrowaveIsaacFactory(IsaacFactory):
    @staticmethod    
    def load_hand_asset(obj, whichHand):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), obj.cfg["env"]["asset"]["handAssetRoot"])
        if whichHand == "left":
            hand_asset_file = obj.cfg["env"]["asset"]["leftHandAssetFileName"]
        elif whichHand == "right":
            hand_asset_file = obj.cfg["env"]["asset"]["rightHandAssetFileName"]
        else:
            print("Error: enter either left or right as hand argument")
            exit()
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = obj.cfg["env"]["asset"]["handAssetOptions"]["fixBaseLink"]
        asset_options.disable_gravity = obj.cfg["env"]["asset"]["handAssetOptions"]["disableGravity"]
        asset_options.flip_visual_attachments = False
        asset_options.collapse_fixed_joints = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if obj.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        hand_asset = obj.gym.load_asset(obj.sim, asset_root, hand_asset_file, asset_options)
        num_hand_bodies = obj.gym.get_asset_rigid_body_count(hand_asset)
        num_hand_shapes = obj.gym.get_asset_rigid_shape_count(hand_asset)
        num_hand_dofs = obj.gym.get_asset_dof_count(hand_asset)
        
        return hand_asset, num_hand_bodies, num_hand_shapes, num_hand_dofs

    @staticmethod
    def load_table_assets(obj):
        table_dims = gymapi.Vec3(1.5, 1.5, 0.6)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = obj.gym.create_box(obj.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
    
        return table_asset, table_dims
    
    @staticmethod
    def load_cube_asset(obj):
        object_asset_root = obj.cfg["env"]["asset"]["cubeAssetRoot"]
        object_file_name = obj.cfg["env"]["asset"]["cubeAssetFileName"]
        
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 500
        object_asset_options.use_mesh_materials = True
        object_asset_options.thickness = 1 # to prevent collisions
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        object_asset_options.fix_base_link = False

        object_asset = obj.gym.load_asset(obj.sim, object_asset_root, object_file_name, object_asset_options)
        return object_asset
    
    @staticmethod
    def load_microwave_asset(obj):
        microwave_asset_root = obj.cfg["env"]["asset"]["microwaveAssetRoot"]
        microwave_file_name = obj.cfg["env"]["asset"]["microwaveAssetFileName"]
        
        microwave_asset_options = gymapi.AssetOptions()
        microwave_asset_options.density = 500
        microwave_asset_options.use_mesh_materials = True
        microwave_asset_options.thickness = 1 # to prevent collisions
        microwave_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        microwave_asset_options.fix_base_link = True

        microwave_asset = obj.gym.load_asset(obj.sim, microwave_asset_root, microwave_file_name, microwave_asset_options)
        
        obj.num_microwave_bodies = obj.gym.get_asset_rigid_body_count(microwave_asset)
        obj.num_microwave_shapes = obj.gym.get_asset_rigid_shape_count(microwave_asset)

        obj.num_microwave_dofs = obj.gym.get_asset_dof_count(microwave_asset)
        microwave_dof_props = obj.gym.get_asset_dof_properties(microwave_asset)

        obj.microwave_dof_lower_limits = []
        obj.microwave_dof_upper_limits = []

        for i in range(obj.num_microwave_dofs):
            obj.microwave_dof_lower_limits.append(microwave_dof_props['lower'][i])
            obj.microwave_dof_upper_limits.append(microwave_dof_props['upper'][i])

        obj.microwave_dof_lower_limits = to_torch(obj.microwave_dof_lower_limits, device=obj.device)
        obj.microwave_dof_upper_limits = to_torch(obj.microwave_dof_upper_limits, device=obj.device)

        return microwave_asset
    
    @staticmethod
    def initialize_poses(table_dims):
        left_hand_start_pose = gymapi.Transform()
        left_hand_start_pose.p = gymapi.Vec3(0.2, -0.1, 1.45)
        left_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.5652925671162337, 0, 0.227)

        right_hand_start_pose = gymapi.Transform()
        right_hand_start_pose.p = gymapi.Vec3(0.2, 0.1, 1.45) 
        right_hand_start_pose.r = gymapi.Quat().from_euler_zyx(-1.5652925671162337, 0, -0.227)

        microwave_start_pose = gymapi.Transform()
        microwave_start_pose.p = gymapi.Vec3(-0.8, -0.45, 0.8)
        microwave_start_pose.r = gymapi.Quat().from_euler_zyx(3.141592, 3.141592, 0)

        cube_start_pose = gymapi.Transform()
        cube_start_pose.p = gymapi.Vec3(-0.5, 0.4, 0.575)
        cube_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(-0.8, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        return table_pose, microwave_start_pose, cube_start_pose, left_hand_start_pose, right_hand_start_pose
    
    @staticmethod
    def env_append(obj, env_ptr, left_hand_start_pose, right_hand_start_pose, microwave_start_pose, cube_start_pose, left_hand_actor, right_hand_actor, table_actor, microwave_actor, cube_actor):
        obj.envs.append(env_ptr)
        obj.left_hand_init_states.append([   left_hand_start_pose.p.x,
                                        left_hand_start_pose.p.y,
                                        left_hand_start_pose.p.z,
                                        left_hand_start_pose.r.x,
                                        left_hand_start_pose.r.y,
                                        left_hand_start_pose.r.z,
                                        left_hand_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        obj.right_hand_init_states.append([   right_hand_start_pose.p.x,
                                        right_hand_start_pose.p.z,
                                        right_hand_start_pose.p.y,
                                        right_hand_start_pose.r.x,
                                        right_hand_start_pose.r.y,
                                        right_hand_start_pose.r.z,
                                        right_hand_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        obj.microwave_init_states.append([ microwave_start_pose.p.x, 
                                        microwave_start_pose.p.y, 
                                        microwave_start_pose.p.z,
                                        microwave_start_pose.r.x, 
                                        microwave_start_pose.r.y, 
                                        microwave_start_pose.r.z, 
                                        microwave_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        obj.cube_init_states.append([ cube_start_pose.p.x, 
                                        cube_start_pose.p.y, 
                                        cube_start_pose.p.z,
                                        cube_start_pose.r.x, 
                                        cube_start_pose.r.y, 
                                        cube_start_pose.r.z, 
                                        cube_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        left_hand_idx = obj.gym.get_actor_index(env_ptr, left_hand_actor, gymapi.DOMAIN_SIM)
        right_hand_idx = obj.gym.get_actor_index(env_ptr, right_hand_actor, gymapi.DOMAIN_SIM)
        table_idx = obj.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
        microwave_idx = obj.gym.get_actor_index(env_ptr, microwave_actor, gymapi.DOMAIN_SIM)
        cube_idx = obj.gym.get_actor_index(env_ptr, cube_actor, gymapi.DOMAIN_SIM)
        
        obj.left_hand_indices.append(left_hand_idx)
        obj.right_hand_indices.append(right_hand_idx)
        obj.table_indices.append(table_idx)
        obj.microwave_indices.append(microwave_idx)
        obj.cube_indices.append(cube_idx)
    
    @staticmethod
    def env_initialisation(obj):
        obj.envs = []
        obj.microwave_init_states = []
        obj.cube_init_states = []
        obj.left_hand_init_states = []
        obj.right_hand_init_states = []
        obj.left_hand_indices = []
        obj.right_hand_indices = []
        obj.microwave_indices = []
        obj.cube_indices = []
        obj.table_indices = []
    
    @staticmethod
    def make_tensors(obj):
        obj.left_hand_init_states = to_torch(obj.left_hand_init_states, device=obj.device).view(obj.num_envs, 13)
        obj.right_hand_init_states = to_torch(obj.right_hand_init_states, device=obj.device).view(obj.num_envs, 13)
        obj.microwave_init_states = to_torch(obj.microwave_init_states, device=obj.device).view(obj.num_envs, 13)
        obj.cube_init_states = to_torch(obj.cube_init_states, device=obj.device).view(obj.num_envs, 13)
        obj.left_hand_indices = to_torch(obj.left_hand_indices, dtype=torch.long, device=obj.device)
        obj.right_hand_indices = to_torch(obj.right_hand_indices, dtype=torch.long, device=obj.device)
        obj.microwave_indices = to_torch(obj.microwave_indices, dtype=torch.long, device=obj.device)
        obj.cube_indices = to_torch(obj.cube_indices, dtype=torch.long, device=obj.device)
        obj.lower_limits = torch.tensor(obj.lower_limits, device=obj.device)
        obj.upper_limits = torch.tensor(obj.upper_limits, device=obj.device)

class Microwave(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = CustomConfigFactory.initialize_config(cfg, device_type, device_id, headless)
        
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.dt = self.sim_params.dt
        
        CustomConfigFactory.initialize_task_params(self, cfg)
        
        super().__init__(cfg=self.cfg)
        
        CustomConfigFactory.setup_camera(self)
        CustomConfigFactory.setup_tensors(self)
        CustomConfigFactory.setup_hand_positions(self)
        CustomConfigFactory.setup_microwave_position(self)
        CustomConfigFactory.setup_dataset(self)
        CustomConfigFactory.setup_stages(self)

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
        
        left_hand_asset, self.num_hand_bodies, self.num_hand_shapes, self.num_hand_dofs = MicrowaveIsaacFactory.load_hand_asset(self, "left")
        right_hand_asset, self.num_hand_bodies, self.num_hand_shapes, self.num_hand_dofs = MicrowaveIsaacFactory.load_hand_asset(self, "right")
        table_asset, table_dims = MicrowaveIsaacFactory.load_table_assets(self)
        microwave_asset = MicrowaveIsaacFactory.load_microwave_asset(self)
        cube_asset = MicrowaveIsaacFactory.load_cube_asset(self)

        robot_dof_props, self.lower_limits, self.upper_limits = MicrowaveIsaacFactory.initialize_robot_dof_props(self.gym, left_hand_asset)

        table_pose, microwave_start_pose, cube_start_pose, left_hand_start_pose, right_hand_start_pose = MicrowaveIsaacFactory.initialize_poses(table_dims)

        MicrowaveIsaacFactory.env_initialisation(self)
        
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.num_hand_bodies + 100, self.num_hand_shapes + 100, True)

            left_hand_actor = self.gym.create_actor(env_ptr, left_hand_asset, left_hand_start_pose, "left_hand", i, -1, 0)
            right_hand_actor = self.gym.create_actor(env_ptr, right_hand_asset, right_hand_start_pose, "right_hand", i, 0, 0)
            
            self.gym.set_actor_dof_properties(env_ptr, left_hand_actor, robot_dof_props)
            self.gym.set_actor_dof_properties(env_ptr, right_hand_actor, robot_dof_props)
            
            microwave_actor = self.gym.create_actor(env_ptr, microwave_asset, microwave_start_pose, "microwave", i, 0, 0)     
            self.gym.set_actor_scale(env_ptr, microwave_actor, 0.45)
            
            cube_actor = self.gym.create_actor(env_ptr, cube_asset, cube_start_pose, "cube", i, 0, 0)
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 0)
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            MicrowaveIsaacFactory.env_append(self, env_ptr, left_hand_start_pose, right_hand_start_pose, microwave_start_pose, cube_start_pose, left_hand_actor, right_hand_actor, table_actor, microwave_actor, cube_actor)

        MicrowaveIsaacFactory.make_tensors(self)
    
    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf,
            self.max_episode_length, self.cube_pos, self.cube_pos, self.microwave_handle_pos, self.microwave_handle_rot,
            self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.actions, self.action_penalty_scale
        )
        # self.extras['successes'] = self.successes
        # self.extras['consecutive_successes'] = self.consecutive_successes

    def compute_observations(self):
        MicrowaveIsaacFactory.refresh_tensors(self)

        self.cube_pose = self.root_state_tensor[self.cube_indices, 0:7]
        self.cube_pos = self.root_state_tensor[self.cube_indices, 0:3]
        self.cube_rot = self.root_state_tensor[self.cube_indices, 3:7]
        
        self.microwave_handle_pos = self.rigid_body_states[:, 54, 0:3] # 1: base, 2: link1, 3: link0
        self.microwave_handle_rot = self.rigid_body_states[:, 54, 3:7]
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.2)
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.4)
        self.microwave_handle_pos = self.microwave_handle_pos + quat_apply(self.microwave_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.075)

        self.left_hand_pos = self.rigid_body_states[:, 7, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 7, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0)
        
        self.right_hand_pos = self.rigid_body_states[:, 33, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 33, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * -0.15)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0)

        MicrowaveIsaacFactory.make_observations(self, "left_hand_ff", 11)
        MicrowaveIsaacFactory.make_observations(self, "left_hand_mf", 19)
        MicrowaveIsaacFactory.make_observations(self, "left_hand_rf", 23)
        MicrowaveIsaacFactory.make_observations(self, "left_hand_lf", 15)
        MicrowaveIsaacFactory.make_observations(self, "left_hand_th", 27)
        
        # MicrowaveIsaacFactory.add_displacement(self, "left_hand_ff", axis="x", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "left_hand_mf", axis="x", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "left_hand_rf", axis="x", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "left_hand_lf", axis="x", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "left_hand_th", axis="x", dist=0.03)
        
        MicrowaveIsaacFactory.make_observations(self, "right_hand_ff", 11+27)
        MicrowaveIsaacFactory.make_observations(self, "right_hand_mf", 19+27)
        MicrowaveIsaacFactory.make_observations(self, "right_hand_rf", 23+27)
        MicrowaveIsaacFactory.make_observations(self, "right_hand_lf", 15+27)
        MicrowaveIsaacFactory.make_observations(self, "right_hand_th", 23)
        
        # MicrowaveIsaacFactory.add_displacement(self, "right_hand_ff", axis="y", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "right_hand_mf", axis="y", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "right_hand_rf", axis="y", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "right_hand_lf", axis="y", dist=0.03)
        # MicrowaveIsaacFactory.add_displacement(self, "right_hand_th", axis="y", dist=0.03)

        self.compute_full_state()
        
    def compute_full_state(self, asymm_obs=False):
        # left hand joint positions and velocities
        self.obs_buf[:, :27] = unscale(self.left_hand_dof_pos, self.lower_limits, self.upper_limits)
        self.obs_buf[:, 27:54] = 0.2 * self.left_hand_dof_vel
        
        # left hand actions
        action_obs_start = 54
        self.obs_buf[:, 54:67] = self.actions[:, :13]
        
        # right hand joint positions and velocities
        self.obs_buf[:, 67:94] = unscale(self.right_hand_dof_pos, self.lower_limits, self.upper_limits)
        self.obs_buf[:, 94:121] = 0.2 * self.right_hand_dof_vel
        
        # right hand actions
        self.obs_buf[:, 121:134] = self.actions[:, 13:26]

        # microwave 3d position and rotation
        self.obs_buf[:, 134:137] = self.microwave_handle_pos
        self.obs_buf[:, 137:141] = self.microwave_handle_rot
        
        # cube 3d position and rotation
        self.obs_buf[:, 141:144] = self.cube_pos
        self.obs_buf[:, 144:148] = self.cube_rot

    def reset(self, env_ids, goal_env_ids):
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_dofs * 2 + 1), device=self.device)
        
        # reset microwave dof
        microwave_pos = self.microwave_default_dof_pos
        self.microwave_dof_pos[env_ids, :] = microwave_pos
        self.prev_targets[env_ids, 54:] = microwave_pos
        self.cur_targets[env_ids, 54:] = microwave_pos

        # reset cube pos
        self.root_state_tensor[self.cube_indices[env_ids]] = self.cube_init_states[env_ids].clone()
        self.root_state_tensor[self.cube_indices[env_ids], 0:2] = self.cube_init_states[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.cube_indices[env_ids], self.up_axis_idx] = self.cube_init_states[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]
        self.root_state_tensor[self.cube_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.cube_indices[env_ids], 7:13])
      
        # hand delta computation
        left_delta_max = self.upper_limits - self.left_hand_default_pos
        left_delta_min = self.lower_limits - self.left_hand_default_pos
        left_rand_delta = left_delta_min + (left_delta_max - left_delta_min) * rand_floats[:, :27]
        
        right_delta_max = self.upper_limits - self.right_hand_default_pos
        right_delta_min = self.lower_limits - self.right_hand_default_pos
        right_rand_delta = right_delta_min + (right_delta_max - right_delta_min) * rand_floats[:, 27:54]
        
        # reset left_hand
        left_pos = self.left_hand_default_pos + self.reset_dof_pos_noise * left_rand_delta

        self.left_hand_dof_pos[env_ids, :] = left_pos
        self.prev_targets[env_ids, :27] = left_pos
        self.cur_targets[env_ids, :27] = left_pos

        left_hand_indices = self.left_hand_indices[env_ids].to(torch.int32)

        # reset right_hand
        right_pos = self.right_hand_default_pos + self.reset_dof_pos_noise * right_rand_delta

        self.right_hand_dof_pos[env_ids, :] = right_pos
        self.prev_targets[env_ids, 27:54] = right_pos
        self.cur_targets[env_ids, 27:54] = right_pos

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
        self.left_grasp_success[env_ids] = 0
        self.right_grasp_success[env_ids] = 0
        self.left_open_success[env_ids] = 0
        self.right_lift_success[env_ids] = 0
        self.right_reach_success[env_ids] = 0

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
        
        self.cur_targets[:, [8,12,16,20,24]] = scale(self.actions[:, 3:8], self.lower_limits[[8,12,16,20,24]], self.upper_limits[[8,12,16,20,24]])
        self.cur_targets[:, [8,12,16,20,24]] = self.act_avg * self.cur_targets[:, [8,12,16,20,24]] + (1.0 - self.act_avg) * self.prev_targets[:, [8,12,16,20,24]]
        self.cur_targets[:, [9,13,17,21,25]] = self.cur_targets[:, [10,14,18,22,26]] = scale(self.actions[:, 8:13], self.lower_limits[[9,13,17,21,25]], self.upper_limits[[9,13,17,21,25]])
        self.cur_targets[:, [9,13,17,21,25]] = self.cur_targets[:, [10,14,18,22,26]] = self.act_avg * self.cur_targets[:, [9,13,17,21,25]] + (1.0 - self.act_avg) * self.prev_targets[:, [9,13,17,21,25]]
        
        self.cur_targets[:, [35,39,43,47,51]] = scale(self.actions[:, 16:21], self.lower_limits[[8,12,16,20,24]], self.upper_limits[[8,12,16,20,24]])
        self.cur_targets[:, [35,39,43,47,51]] = self.act_avg * self.cur_targets[:, [35,39,43,47,51]] + (1.0 - self.act_avg) * self.prev_targets[:, [35,39,43,47,51]]
        self.cur_targets[:, [36,40,44,48,52]] = self.cur_targets[:, [37,41,45,49,53]] = scale(self.actions[:, 21:26], self.lower_limits[[9,13,17,21,25]], self.upper_limits[[9,13,17,21,25]])
        self.cur_targets[:, [36,40,44,48,52]] = self.cur_targets[:, [37,41,45,49,53]] = self.act_avg * self.cur_targets[:, [36,40,44,48,52]] + (1.0 - self.act_avg) * self.prev_targets[:, [36,40,44,48,52]]

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
        l_delta = control_ik(self.left_jacobian[:, 7 - 1, :, :7], self.device, l_dpose, self.num_envs)
        r_dpose = torch.cat([right_pos_err, right_rot_err], -1).unsqueeze(-1)
        r_delta = control_ik(self.right_jacobian[:, 7 - 1, :, :7], self.device, r_dpose, self.num_envs)
        
        self.cur_targets[:, :7] = self.left_hand_dof_pos[:, 0:7] + l_delta[:, :7]
        self.cur_targets[:, :7][stage_2_ids] = self.prev_targets[:, :7][stage_2_ids]
        self.cur_targets[:, :7][stage_3_ids] = self.left_trajectory[0][0:7]

        self.cur_targets[:, 27:34] = self.right_hand_dof_pos[:, 0:7] + r_delta[:, :7]
        self.cur_targets[:, 27:34][stage_2_ids] = self.prev_targets[:, 27:34][stage_2_ids]
        self.cur_targets[:, 27:34][stage_3_ids] = self.left_trajectory[1][0:7]
        
        self.cur_targets[:, 7:27][stage_3_ids] = self.prev_targets[:, 7:27][stage_3_ids]
        self.cur_targets[:, 34:54][stage_3_ids] = self.prev_targets[:, 34:54][stage_3_ids]

        self.cur_targets[:, :27] = tensor_clamp(self.cur_targets[:, :27], self.lower_limits, self.upper_limits)
        self.cur_targets[:, 27:54] = tensor_clamp(self.cur_targets[:, 27:54], self.lower_limits, self.upper_limits)
        self.prev_targets[:,:] = self.cur_targets[:,:]

        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        
        if self.viewer and self.debug:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                MicrowaveIsaacFactory.add_debug_lines(self, self.envs[i], self.microwave_handle_pos[i], self.microwave_handle_rot[i])
                # MicrowaveIsaacFactory.add_debug_lines(self, self.envs[i], self.left_hand_mf_pos[i], self.left_hand_mf_rot[i])
                # MicrowaveIsaacFactory.add_debug_lines(self, self.envs[i], self.left_hand_rf_pos[i], self.left_hand_rf_rot[i])
                # MicrowaveIsaacFactory.add_debug_lines(self, self.envs[i], self.left_hand_lf_pos[i], self.left_hand_lf_rot[i])
                # MicrowaveIsaacFactory.add_debug_lines(self, self.envs[i], self.left_hand_th_pos[i], self.left_hand_th_rot[i])
       
@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf,
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
    
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    
    return reward, resets, goal_resets, progress_buf

