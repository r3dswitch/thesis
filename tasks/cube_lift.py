from tasks.hand_base.base_task import BaseTask

from .Math_Helpers import *
from .Isaac_Creator import *
from .Init_Creator import *

class CubeConfigFactory(ConfigFactory):
    @staticmethod
    def setup_tensors(obj):
        super(CubeConfigFactory, CubeConfigFactory).setup_tensors(obj)
        obj.grasp_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.lift_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.successes = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        # obj.target_euler = to_torch([0.0, 0.0, 0.0], device=obj.device).repeat((obj.num_envs, 1))
        obj.target_euler = to_torch([0.0, -1.57, 0.0], device=obj.device).repeat((obj.num_envs, 1))
        
class CubeIsaacFactory(IsaacFactory):
    @staticmethod
    def initialize_poses():
        table1_pose = gymapi.Transform()
        table1_pose.p = gymapi.Vec3(0.0, 0, 0.25)
        table1_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
            
        table2_pose = gymapi.Transform()
        table2_pose.p = gymapi.Vec3(0.3, -0.25, 0.5)
        table2_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.3, -0.25, 0.8)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0,0,0)

        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(-0.4, 0.3, 0.5)  # Based on table height
        hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        return table1_pose, table2_pose, object_start_pose, hand_start_pose

class CubeLift(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, 
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    
        self.cfg = CubeConfigFactory.initialize_config(cfg, device_type, device_id, headless)
        
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.dt = self.sim_params.dt
        
        self, self.cfg = CubeConfigFactory.initialize_task_params(self, self.cfg)
        
        super().__init__(cfg=self.cfg)
        
        CubeConfigFactory.setup_camera(self) # self, cam pos, cam target
        CubeConfigFactory.setup_tensors(self) # self, jacobian_actor_name, end_effector_link_name, wrist_rot_mod_quaternion
        CubeConfigFactory.setup_hand_positions(self) # self, init_pos_list, trajectory_list_of_lists
        CubeConfigFactory.setup_dataset(self) # self, base_dir(validated), mesh_dir(manifold), dataset_regex, grasp_path(filtered_grasps.npy), is_full_dataset
        CubeConfigFactory.setup_stages(self) # self, prempt, ik, grasp, rl, lift
    
    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        self.sim_params.gravity.z = -9.8 
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        hand_asset, self.num_hand_bodies, self.num_hand_shapes, self.num_hand_dofs = CubeIsaacFactory.load_hand_asset(self)
        table_asset, table_asset_2, table_dims = CubeIsaacFactory.load_table_assets(self)
        object_asset = CubeIsaacFactory.load_object_asset(self)

        robot_dof_props, self.lower_limits, self.upper_limits = CubeIsaacFactory.initialize_robot_dof_props(self.gym, hand_asset)

        table1_pose, table2_pose, object_start_pose, hand_start_pose = CubeIsaacFactory.initialize_poses()

        CubeIsaacFactory.env_initialisation(self)
        
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.num_hand_bodies + 100, self.num_hand_shapes + 100, True)

            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, robot_dof_props)

            table1_actor = self.gym.create_actor(env_ptr, table_asset, table1_pose, "table", i, -1, 0)
            table2_actor = self.gym.create_actor(env_ptr, table_asset_2, table2_pose, "table2", i, -1, 0)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 1)
            
            self.gym.set_rigid_body_color(env_ptr, table2_actor, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0.3,0.1,0.1))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            CubeIsaacFactory.env_append(self, env_ptr, hand_start_pose, object_start_pose, hand_actor, table1_actor, table2_actor, object_actor)
        
        CubeIsaacFactory.make_tensors(self)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.grasp_success[:], self.lift_success[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.grasp_success, self.lift_success, self.consecutive_successes, 
            self.max_episode_length, self.object_init_pos, self.object_pos, self.wrist_pos, self.hand_ff_pos, self.hand_rf_pos, self.hand_mf_pos, self.hand_th_pos, self.hand_lf_pos,
            self.actions, self.max_consecutive_successes, self.av_factor, self.rl_start
        )

        self.extras['Grasp Success'] = self.grasp_success
        self.extras['Lift Success'] = self.lift_success
        self.extras['Consecutive Successes'] = self.consecutive_successes

    def compute_observations(self):
        CubeIsaacFactory.refresh_tensors(self)

        CubeIsaacFactory.make_observations(self, "wrist", self.wrist_rigid_body_index)
        # self.wrist_rot = quat_mul(self.wrist_rot, self.wrist_rot_mod)
        CubeIsaacFactory.add_displacement(self, "wrist", axis="z", dist=-0.15)
        
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7] 
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        CubeIsaacFactory.make_observations(self, "hand_ff", 11)
        CubeIsaacFactory.make_observations(self, "hand_mf", 19)
        CubeIsaacFactory.make_observations(self, "hand_rf", 23)
        CubeIsaacFactory.make_observations(self, "hand_lf", 15)
        CubeIsaacFactory.make_observations(self, "hand_th", 27)
        
        CubeIsaacFactory.add_displacement(self, "hand_ff", axis="y", dist=0.03)
        CubeIsaacFactory.add_displacement(self, "hand_mf", axis="y", dist=0.03)
        CubeIsaacFactory.add_displacement(self, "hand_rf", axis="y", dist=0.03)
        CubeIsaacFactory.add_displacement(self, "hand_lf", axis="y", dist=0.03)
        CubeIsaacFactory.add_displacement(self, "hand_th", axis="y", dist=0.03)
        
        self.compute_full_state()
    
    def compute_full_state(self):
        self.obs_buf[:, :27] = unscale(self.hand_dof_pos, self.lower_limits, self.upper_limits)
        self.obs_buf[:, 27:54] = 0.2 * self.hand_dof_vel
        self.obs_buf[:, 54:64] = self.actions[:, :10]
        self.obs_buf[:, 64:71] = self.object_pose
        self.obs_buf[:, 71:74] = self.object_linvel
        self.obs_buf[:, 74:77] = 0.2 * self.object_angvel
                    
    def reset_idx(self, env_ids, goal_env_ids):
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(-1.0,1.0, (len(env_ids),self.num_hand_dofs*2), device=self.device)
        
        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_states[env_ids].clone()

        degree_one_side = 0
        new_object_rot = quat_mul(self.object_init_states[env_ids, 3:7], randomize_rotation(rand_floats[:, 3], self.y_unit_tensor[env_ids], degree_one_side))
        
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_states[env_ids, 0:2] 
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_states[env_ids, self.up_axis_idx] 
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        
        # reset hand
        delta_max = self.upper_limits - self.hand_dof_default_pos
        delta_min = self.lower_limits - self.hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_hand_dofs]

        pos = self.hand_dof_default_pos + self.reset_dof_pos_noise * rand_delta

        self.hand_dof_pos[env_ids, :] = pos
        
        hand_indices = self.hand_indices[env_ids].to(torch.int32)

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.prev_targets[env_ids, :self.num_hand_dofs] = pos 
        self.cur_targets[env_ids, :self.num_hand_dofs] = pos 

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
        self.lift_success[env_ids] = 0

    def pre_physics_step(self, actions):
        CubeIsaacFactory.refresh_tensors(self)
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        
        """
        Stage 1: Arm IK with open hand to intermediate position
        Stage 2: Arm IK with open hand to grasp position
        Stage 3: Fixed Arm with finger IK to default position
        Stage 4: Finger RL
        Stage 5: Arm Lift
        """
        reach_ids = [self.progress_buf > 0]
        grasp_ids = [self.progress_buf > 50]
        lift_ids = [self.progress_buf > 100]

        pos_err = self.actions[:, 10:13] * 0.2
        # pos_err[:, 2][reach_ids] += 0.05

        target_rot = quat_from_euler_xyz(self.target_euler[:, 0], self.target_euler[:, 1], self.target_euler[:, 2])
        rot_err = orientation_error(target_rot, self.rigid_body_states[:, 7, 3:7].clone()) * 5
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_tensor[:, 7 - 1, :, :7], self.device, dpose, self.num_envs)
        
        self.cur_targets[:, :7][reach_ids] = self.hand_dof_pos[:, 0:7][reach_ids] + delta[:, :7][reach_ids]
        self.cur_targets[:, :7][grasp_ids] = self.prev_targets[:, :7][grasp_ids]
        self.cur_targets[:, :7][lift_ids] = self.trajectory_list[0][:7]

        self.cur_targets[:, 7:27][reach_ids] = 0.01
        
        # Stage 3 Finger Control
        a = scale(torch.zeros_like(self.actions[:, :5]), self.lower_limits[[7,11,15,19,23]], self.upper_limits[[7,11,15,19,23]])
        b = scale(self.actions[:, :5], self.lower_limits[[8,12,16,20,24]], self.upper_limits[[8,12,16,20,24]]) * 0.5
        c = scale(self.actions[:, 5:10], self.lower_limits[[9,13,17,21,25]], self.upper_limits[[9,13,17,21,25]])  * 0.5
        d = scale(self.actions[:, 5:10], self.lower_limits[[10,14,18,22,26]], self.upper_limits[[10,14,18,22,26]])  * 0.5

        stacked = torch.stack([a, b, c, d], dim=-1)
        interleaved = stacked.reshape(stacked.shape[0], -1)
        
        self.cur_targets[:, 7:27][grasp_ids] = interleaved[grasp_ids]
        self.cur_targets[:, 7:27][lift_ids] = self.prev_targets[:, 7:27][lift_ids]
        
        self.cur_targets = self.cur_targets * self.act_avg + self.prev_targets * (1 - self.act_avg)
        self.cur_targets[:, :27] = tensor_clamp(self.cur_targets[:, :27], self.lower_limits, self.upper_limits)
        
        self.prev_targets = self.cur_targets
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        self.debug = False
        if self.debug:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            for i in range(self.num_envs):
                CubeIsaacFactory.add_debug_lines(self, self.envs[i], self.object_pos[i], self.object_rot[i])
                CubeIsaacFactory.add_debug_lines(self, self.envs[i], self.wrist_pos[i], self.wrist_rot[i])

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, grasp_success, lift_success, consecutive_successes,
    max_episode_length: float, object_init_pos, object_pos, wrist_pos, hand_ff_pos, hand_rf_pos, hand_mf_pos, hand_th_pos, hand_lf_pos,
    actions, max_consecutive_successes: int, av_factor: float, rl_start: int
):
    hand_finger_dist = (torch.norm(object_pos - hand_ff_pos, p=2, dim=-1) + torch.norm(object_pos - hand_mf_pos, p=2, dim=-1)
                            + torch.norm(object_pos - hand_rf_pos, p=2, dim=-1) + torch.norm(object_pos - hand_th_pos, p=2, dim=-1) + torch.norm(object_pos - hand_lf_pos, p=2, dim=-1))

    grasp_reward = torch.exp(- 1 * (hand_finger_dist))

    lift_reward = torch.exp((object_pos[:, 2] - object_init_pos[:, 2]))
    
    resets = torch.where(hand_finger_dist >= 2.5, torch.ones_like(reset_buf), reset_buf)
    
    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    reward = lift_reward + grasp_reward
    
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(lift_success * resets.float())
    
    grasp_success = torch.where(grasp_success == 0, torch.where((grasp_reward > 0.5), torch.ones_like(grasp_success), grasp_success), grasp_success)
    lift_success = torch.where(lift_success == 0, torch.where((lift_reward > 1), torch.ones_like(lift_success), lift_success), lift_success)
    
    cons_successes = torch.where(   num_resets > 0, 
                                    av_factor * finished_cons_successes / num_resets + 
                                    (1.0 - av_factor) * consecutive_successes, 
                                    consecutive_successes   )

    return reward, resets, reset_goal_buf, progress_buf, grasp_success, lift_success, cons_successes
