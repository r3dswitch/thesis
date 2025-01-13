from tasks.hand_base.base_task import BaseTask

from .Math_Helpers import *
from .Isaac_Creator import *
from .Init_Creator import *

class CustomConfigFactory(ConfigFactory):
    @staticmethod
    def setup_tensors(obj):
        super(CustomConfigFactory, CustomConfigFactory).setup_tensors(obj)
        obj.grasp_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.move_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.orientation_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.desired_pos = obj.root_state_tensor[obj.hand_indices, 0:3].clone()
        obj.desired_rot = obj.root_state_tensor[obj.hand_indices, 3:7].clone()
        obj.hammer_head_pos = obj.root_state_tensor[obj.hand_indices, 0:3].clone()
        obj.hammer_head_rot = obj.root_state_tensor[obj.hand_indices, 3:7].clone()
        
    @staticmethod
    def setup_hand_positions(obj):
        init_pos = obj.cfg["env"]["handPositions"]["defaultPos"]
        traj_list = obj.cfg["env"]["handPositions"]["trajectoryPosList"]
        
        obj.hand_dof_default_pos = torch.zeros(
            obj.num_hand_dofs, dtype=torch.float, device=obj.device
        )
        obj.hand_dof_default_pos[:7] = torch.tensor( init_pos, dtype=torch.float, device=obj.device )
        obj.hand_dof_default_pos[7:27] = torch.tensor( [0,0.6,0.6,0,0,0.6,0.6,0,0,0.6,0.6,0,0,0.6,0.6,0,0,0.6,0.6,0], dtype=torch.float, device=obj.device )

   
class HammerIsaacFactory(IsaacFactory):
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
    def load_object_asset(obj):
        object_asset_root = obj.cfg["env"]["asset"]["objectAssetRoot"]
        object_file_name = obj.cfg["env"]["asset"]["objectAssetFileName"]
        
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = False
        object_asset_options.vhacd_enabled = True
        object_asset_options.armature = 0.025
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        object_asset = obj.gym.load_asset(obj.sim, object_asset_root, object_file_name, object_asset_options)
        return object_asset
    
    @staticmethod
    def initialize_poses(table_dims):
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.19, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.3,0.3,1)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0.0, 0.785)

        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(-0.35, 0, 0.6) 
        hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)
        
        return table_pose, object_start_pose, hand_start_pose
    
    @staticmethod
    def env_append(self, env_ptr, hand_start_pose, object_start_pose, hand_actor, table_actor, object_actor):
        self.envs.append(env_ptr)
        
        self.hand_init_states.append([   hand_start_pose.p.x,
                                        hand_start_pose.p.y,
                                        hand_start_pose.p.z,
                                        hand_start_pose.r.x,
                                        hand_start_pose.r.y,
                                        hand_start_pose.r.z,
                                        hand_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        self.object_init_states.append([ object_start_pose.p.x, 
                                        object_start_pose.p.y, 
                                        object_start_pose.p.z,
                                        object_start_pose.r.x, 
                                        object_start_pose.r.y, 
                                        object_start_pose.r.z, 
                                        object_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        table_idx = self.gym.get_actor_index(env_ptr, table_actor, gymapi.DOMAIN_SIM)
        object_idx = self.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
        
        self.hand_indices.append(hand_idx)
        self.table_indices.append(table_idx)
        self.object_indices.append(object_idx)
    
class HammerOrient(BaseTask):

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
        CustomConfigFactory.setup_dataset(self)
        CustomConfigFactory.setup_stages(self)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        self.sim_params.gravity.z = -7 #-2

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

        hand_asset, self.num_hand_bodies, self.num_hand_shapes, self.num_hand_dofs = HammerIsaacFactory.load_hand_asset(self)
        table_asset, table_dims = HammerIsaacFactory.load_table_assets(self)
        object_asset = HammerIsaacFactory.load_object_asset(self)

        robot_dof_props, self.lower_limits, self.upper_limits = HammerIsaacFactory.initialize_robot_dof_props(self.gym, hand_asset)

        table_pose, object_start_pose, hand_start_pose = HammerIsaacFactory.initialize_poses(table_dims)

        HammerIsaacFactory.env_initialisation(self)
        
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.num_hand_bodies + 100, self.num_hand_shapes + 100, True)

            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, robot_dof_props)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 1)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            HammerIsaacFactory.env_append(self, env_ptr, hand_start_pose, object_start_pose, hand_actor, table_actor, object_actor)
        
        HammerIsaacFactory.make_tensors(self)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.grasp_success[:], self.move_success[:], self.orientation_success[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.grasp_success, self.move_success, self.orientation_success, self.consecutive_successes, 
            self.max_episode_length, self.object_pos, self.object_rot, self.hammer_head_pos, self.hammer_head_rot, self.desired_pos, self.desired_rot, self.hand_ff_pos, self.hand_rf_pos, self.hand_mf_pos, self.hand_th_pos,
            self.actions, self.max_consecutive_successes, self.av_factor, self.x_unit_tensor, self.y_unit_tensor, self.z_unit_tensor, self.num_envs
        )
        
        self.extras['grasp_success'] = self.grasp_success
        self.extras['move_success'] = self.move_success
        self.extras['orientation_success'] = self.orientation_success

    def compute_observations(self, is_searching=False, last_action=0):
        HammerIsaacFactory.refresh_tensors(self)

        HammerIsaacFactory.make_observations(self, "wrist", self.wrist_rigid_body_index)
        self.wrist_rot = quat_mul(self.wrist_rot, self.wrist_rot_mod)

        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        HammerIsaacFactory.make_observations(self, "hand_ff", 11)
        HammerIsaacFactory.make_observations(self, "hand_mf", 19)
        HammerIsaacFactory.make_observations(self, "hand_rf", 23)
        HammerIsaacFactory.make_observations(self, "hand_lf", 15)
        HammerIsaacFactory.make_observations(self, "hand_th", 27)
        
        HammerIsaacFactory.add_displacement(self, "hand_ff", axis="y", dist=0.03)
        HammerIsaacFactory.add_displacement(self, "hand_mf", axis="y", dist=0.03)
        HammerIsaacFactory.add_displacement(self, "hand_rf", axis="y", dist=0.03)
        HammerIsaacFactory.add_displacement(self, "hand_lf", axis="y", dist=0.03)
        HammerIsaacFactory.add_displacement(self, "hand_th", axis="y", dist=0.03)
        
        self.hand_finger_dist = (torch.norm(self.object_pos - self.hand_ff_pos, p=2, dim=-1) + torch.norm(self.object_pos - self.hand_mf_pos, p=2, dim=-1)
                                + torch.norm(self.object_pos - self.hand_rf_pos, p=2, dim=-1) + torch.norm(self.object_pos - self.hand_th_pos, p=2, dim=-1))
        
        self.desired_pos = to_torch([0.5,0.5,1], device=self.device).repeat((self.num_envs, 1))
        self.desired_rot = to_torch([0.707,0.707,0,0], device=self.device).repeat((self.num_envs, 1)) 
        
        self.hammer_head_pos, self.hammer_head_rot = self.object_pos, self.object_rot
        HammerIsaacFactory.add_displacement(self, "hammer_head", axis="y", dist=0.250)
        HammerIsaacFactory.add_displacement(self, "hammer_head", axis="z", dist=0.075)
        
    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0,1.0, (len(env_ids),self.num_dofs*2), device=self.device)
        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_states[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_states[env_ids, 0:2] 
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_states[env_ids, self.up_axis_idx] 

        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        # reset arm
        delta_max = self.upper_limits - self.hand_dof_default_pos
        delta_min = self.lower_limits - self.hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_dofs]

        pos = self.hand_dof_default_pos #+ self.reset_dof_pos_noise * rand_delta

        self.hand_dof_pos[env_ids, :] = pos
        
        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.prev_targets[env_ids, :self.num_dofs] = pos #self.hand_dof_pos[env_ids]
        self.cur_targets[env_ids, :self.num_dofs] = pos #self.hand_dof_pos[env_ids]

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.prev_targets),
                                                gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        
        all_indices = torch.unique(torch.cat([hand_indices,
                                              object_indices]).to(torch.int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.grasp_success[env_ids] = 0
        self.move_success[env_ids] = 0
        self.orientation_success[env_ids] = 0

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

        stage_1 = [self.progress_buf > 0] # hold hammer
        stage_2 = [self.progress_buf > 150] # move hammer
        stage_3 = [self.progress_buf > 250] # orient hammer
        
        pos_err = (self.desired_pos - self.wrist_pos) * 0.2
        rot_err = orientation_error(self.desired_rot, self.wrist_rot) * 5
        
        pos_offset = self.hammer_head_pos - self.wrist_pos 
        pos_err -= pos_offset * 0.2
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_tensor[:, 7 - 1, :, :7], self.device, dpose, self.num_envs)
        
        a = scale(torch.zeros_like(self.actions[:, :5]), self.lower_limits[[7,11,15,19,23]], self.upper_limits[[7,11,15,19,23]])
        b = scale(self.actions[:, :5], self.lower_limits[[8,12,16,20,24]], self.upper_limits[[8,12,16,20,24]]) * 0.75
        c = scale(self.actions[:, 5:10], self.lower_limits[[9,13,17,21,25]], self.upper_limits[[9,13,17,21,25]])  * 0.75
        d = scale(self.actions[:, 5:10], self.lower_limits[[10,14,18,22,26]], self.upper_limits[[10,14,18,22,26]])  * 0.75
        stacked = torch.stack([a, b, c, d], dim=-1)
        interleaved = stacked.reshape(stacked.shape[0], -1)
        
        self.cur_targets[:, :7][stage_1] = self.prev_targets[:, :7][stage_1]
        self.cur_targets[:, :7][stage_2] = self.hand_dof_pos[:, 0:7][stage_2] + delta[:, :7][stage_2]
        self.cur_targets[:, :7][stage_3] = self.prev_targets[:, :7][stage_3]
        
        self.cur_targets[:, 7:27][stage_1] = interleaved[stage_1]
        self.cur_targets[:, 7:27][stage_2] = self.prev_targets[:, 7:27][stage_2]
        self.cur_targets[:, 7:27][stage_3] = interleaved[stage_3]
        
        self.cur_targets = self.act_avg * self.cur_targets + (1.0 - self.act_avg) * self.prev_targets

        self.prev_targets[:, :] = self.cur_targets[:, :]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # self.add_debug_lines(self.envs[i], self.wrist_pos[i], self.wrist_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_init_pos[0], self.object_init_rot[0])
                self.add_debug_lines(self.envs[i], self.desired_pos[i], self.desired_rot[i])
                self.add_debug_lines(self.envs[i], self.hammer_head_pos[i], self.hammer_head_rot[i])

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
    rew_buf, reset_buf, reset_goal_buf, progress_buf, grasp_success, move_success, orientation_success, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, hammer_head_pos, hammer_head_rot, desired_pos, desired_rot, hand_ff_pos, hand_rf_pos, hand_mf_pos, hand_th_pos,
    actions, max_consecutive_successes: int, av_factor: float, x_unit_tensor, y_unit_tensor, z_unit_tensor, num_envs: int
):
    hand_finger_dist = (torch.norm(object_pos - hand_ff_pos, p=2, dim=-1) + torch.norm(object_pos - hand_mf_pos, p=2, dim=-1)
                            + torch.norm(object_pos - hand_rf_pos, p=2, dim=-1) + 3 * torch.norm(object_pos - hand_th_pos, p=2, dim=-1))

    grasp_reward = torch.exp(- 1 * (hand_finger_dist))
    
    object_pos_rew = torch.exp( -1 * (torch.norm(hammer_head_pos - desired_pos, p=2, dim=-1)))
    
    des_rot = quat_apply(desired_rot, z_unit_tensor)
    cur_rot = quat_apply(hammer_head_rot, z_unit_tensor)
    dot_z = torch.bmm(des_rot.view(num_envs, 1, 3), cur_rot.view(num_envs, 3, 1)).squeeze(-1).squeeze(-1) 
    align_reward_z = ((torch.sign(dot_z) * dot_z ** 2) + 1) / 2
    
    object_rot_rew = align_reward_z
    
    resets = torch.where(hand_finger_dist >= 3, torch.ones_like(reset_buf), reset_buf)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    reward = grasp_reward * 5 + object_pos_rew + object_rot_rew * 0.5
    
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(grasp_success * resets.float())
    
    grasp_success = torch.where(grasp_success == 0, torch.where(grasp_reward > 0.6, torch.ones_like(grasp_success), grasp_success), grasp_success)
    move_success = torch.where(move_success == 0, torch.where(object_pos_rew > 0.9, torch.ones_like(move_success), move_success), move_success)
    orientation_success = torch.where(orientation_success == 0, torch.where(object_rot_rew > 0.7, torch.ones_like(orientation_success), orientation_success), orientation_success)
    
    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, reset_goal_buf, progress_buf, grasp_success, move_success, orientation_success, cons_successes

