from tasks.hand_base.base_task import BaseTask

from .Init_Helpers import *
from .Math_Helpers import *
from .Isaac_Helpers import *

class YCBPitcher(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, 
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    
        self.cfg = initialize_config(cfg, device_type, device_id, headless)
        
        self.sim_params = sim_params
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.dt = self.sim_params.dt
        
        self, self.cfg = initialize_task_params(self, self.cfg)
        
        super().__init__(cfg=self.cfg)
        
        setup_camera(self) # self, cam pos, cam target
        setup_tensors(self) # self, jacobian_actor_name, end_effector_link_name, wrist_rot_mod_quaternion
        setup_hand_positions(self) # self, init_pos_list, trajectory_list_of_lists
        setup_dataset(self) # self, base_dir(validated), mesh_dir(manifold), dataset_regex, grasp_path(filtered_grasps.npy), is_full_dataset
        setup_stages(self) # self, prempt, ik, grasp, rl, lift
    
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

        hand_asset, self.num_hand_bodies, self.num_hand_shapes, self.num_hand_dofs = load_hand_asset(self)
        table_asset, table_asset_2, table_dims = load_table_assets(self)
        object_asset = load_object_asset(self)

        robot_dof_props, self.lower_limits, self.upper_limits = initialize_robot_dof_props(self.gym, hand_asset)

        table1_pose, table2_pose, object_start_pose, hand_start_pose = initialize_poses(table_dims)

        env_initialisation(self)
        
        for i in range(num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, self.num_hand_bodies + 100, self.num_hand_shapes + 100, True)

            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, -1, 0)
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, robot_dof_props)

            table1_actor = self.gym.create_actor(env_ptr, table_asset, table1_pose, "table", i, -1, 0)
            table2_actor = self.gym.create_actor(env_ptr, table_asset_2, table2_pose, "table2", i, -1, 0)
            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 1)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            env_append(self, env_ptr, hand_start_pose, object_start_pose, hand_actor, table1_actor, table2_actor, object_actor)
        
        make_tensors(self)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.grasp_success[:], self.lift_success[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.grasp_success, self.lift_success, self.consecutive_successes, 
            self.max_episode_length, self.object_init_pos, self.handle_pos, self.wrist_pos, self.hand_ff_pos, self.hand_rf_pos, self.hand_mf_pos, self.hand_th_pos, self.hand_lf_pos,
            self.actions, self.max_consecutive_successes, self.av_factor, self.rl_start, 
            self.knock_dist, self.ik_fail_dist, self.fail_dist,  self.grasp_success_dist, self.lift_success_dist
        )

        self.extras['Grasp Success'] = self.grasp_success
        self.extras['Lift Success'] = self.lift_success
        self.extras['Consecutive Successes'] = self.consecutive_successes

    def compute_observations(self):
        refresh_tensors(self)

        make_observations(self, "wrist", self.wrist_rigid_body_index)
        self.wrist_rot = quat_mul(self.wrist_rot, self.wrist_rot_mod)

        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.grasp_pos, self.grasp_rot = compute_world_pose(self, self.grasp_list[:self.num_envs, 0:9], self.object_rot, self.object_pos)
        
        self.intermediate_pos = self.grasp_pos
        self.intermediate_rot = self.grasp_rot
        add_displacement(self, "intermediate", axis="z", dist=-0.15)
        
        make_observations(self, "hand_ff", 11)
        make_observations(self, "hand_mf", 19)
        make_observations(self, "hand_rf", 23)
        make_observations(self, "hand_lf", 15)
        make_observations(self, "hand_th", 27)
        
        add_displacement(self, "hand_ff", axis="y", dist=0.03)
        add_displacement(self, "hand_mf", axis="y", dist=0.03)
        add_displacement(self, "hand_rf", axis="y", dist=0.03)
        add_displacement(self, "hand_lf", axis="y", dist=0.03)
        add_displacement(self, "hand_th", axis="y", dist=0.03)
        
        self.handle_pos, self.handle_rot = self.object_pos, self.object_rot
        add_displacement(self, "handle", axis="z", dist=0.06)
            
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
        refresh_tensors(self)
        
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
        stage_1_ids = [self.progress_buf > self.prempt_start]
        stage_2_ids = [self.progress_buf > self.ik_start]
        stage_3_ids = [self.progress_buf > self.grasp_start]
        stage_4_ids = [self.progress_buf > self.rl_start]
        stage_5_ids = [self.progress_buf > self.lift_start]

        condition = self.progress_buf < self.ik_start
        condition = condition.unsqueeze(1).repeat(1,3)

        pos_err = torch.where(condition, self.intermediate_pos - self.wrist_pos, (self.grasp_pos - self.wrist_pos) * 0.5 )
        rot_err = torch.where(condition, orientation_error(self.intermediate_rot, self.wrist_rot), orientation_error(self.grasp_rot, self.wrist_rot))
        
        dpose = torch.cat([pos_err, rot_err], -1).unsqueeze(-1)
        delta = control_ik(self.jacobian_tensor[:, 7 - 1, :, :7], self.device, dpose, self.num_envs)
        
        self.cur_targets[:, :7][stage_1_ids] = self.hand_dof_pos[:, 0:7][stage_1_ids] + delta[:, :7][stage_1_ids]
        self.cur_targets[:, :7][stage_2_ids] = self.hand_dof_pos[:, 0:7][stage_2_ids] + delta[:, :7][stage_2_ids]
        self.cur_targets[:, :7][stage_3_ids] = self.prev_targets[:, :7][stage_3_ids]
        self.cur_targets[:, :7][stage_4_ids] = self.prev_targets[:, :7][stage_4_ids]
        self.cur_targets[:, :7][stage_5_ids] = self.trajectory_list[0][:7]

        self.cur_targets[:, 7:27][stage_1_ids] = 0.01
        self.cur_targets[:, 7:27][stage_2_ids] = 0.01
        self.cur_targets[:, 7:27][stage_3_ids] = self.grasp_list[:self.num_envs, 8:28][stage_3_ids]
        # self.cur_targets[:, 7:27][stage_3_ids] = self.trajectory_list[1][:7]

        # Stage 3 Finger Control
        a = scale(torch.zeros_like(self.actions[:, :5]), self.lower_limits[[7,11,15,19,23]], self.upper_limits[[7,11,15,19,23]])
        b = scale(self.actions[:, :5], self.lower_limits[[8,12,16,20,24]], self.upper_limits[[8,12,16,20,24]]) * 0.25
        c = scale(self.actions[:, 5:10], self.lower_limits[[9,13,17,21,25]], self.upper_limits[[9,13,17,21,25]])  * 0.25
        d = scale(self.actions[:, 5:10], self.lower_limits[[10,14,18,22,26]], self.upper_limits[[10,14,18,22,26]])  * 0.25

        stacked = torch.stack([a, b, c, d], dim=-1)
        interleaved = stacked.reshape(stacked.shape[0], -1)
        
        self.cur_targets[:, 7:27][stage_4_ids] = interleaved[stage_4_ids]
        self.cur_targets[:, 7:27][stage_5_ids] = self.prev_targets[:, 7:27][stage_5_ids]
        
        self.cur_targets = self.cur_targets * self.act_avg + self.prev_targets * (1 - self.act_avg)
        self.cur_targets[:, :27] = tensor_clamp(self.cur_targets[:, :27], self.lower_limits, self.upper_limits)
        
        self.prev_targets = self.cur_targets
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.debug:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            
            for i in range(self.num_envs):
                add_debug_lines(self, self.envs[i], self.object_pos[i], self.object_rot[i])
                add_debug_lines(self, self.envs[i], self.grasp_pos[i], self.grasp_rot[i])
                add_debug_lines(self, self.envs[i], self.handle_pos[i], self.handle_rot[i])

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, grasp_success, lift_success, consecutive_successes,
    max_episode_length: float, object_init_pos, handle_pos, wrist_pos, hand_ff_pos, hand_rf_pos, hand_mf_pos, hand_th_pos, hand_lf_pos,
    actions, max_consecutive_successes: int, av_factor: float, rl_start: int, 
    knock_dist: float, ik_fail_dist: float, fail_dist: float,  grasp_success_dist: float, lift_success_dist: float
):
    hand_finger_dist = (torch.norm(handle_pos - hand_ff_pos, p=2, dim=-1) + torch.norm(handle_pos - hand_mf_pos, p=2, dim=-1)
                            + torch.norm(handle_pos - hand_rf_pos, p=2, dim=-1) + torch.norm(handle_pos - hand_th_pos, p=2, dim=-1) + torch.norm(handle_pos - hand_lf_pos, p=2, dim=-1))

    grasp_reward = torch.exp(- 1 * (hand_finger_dist))
    
    obj_move_dist = torch.norm(handle_pos - object_init_pos, p=2, dim=-1)

    lift_reward = (handle_pos[:, 2] - object_init_pos[:, 2]) * 10
    
    resets = torch.where(hand_finger_dist >= fail_dist, torch.ones_like(reset_buf), reset_buf)
    
    condition = torch.logical_and( obj_move_dist > knock_dist, progress_buf < rl_start) 
    resets = torch.where(condition, torch.ones_like(resets), resets)
    
    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    reward = lift_reward + grasp_reward
    
    if max_consecutive_successes > 0:
        reward = torch.where(timed_out, reward, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(lift_success * resets.float())
    
    grasp_success = torch.where(grasp_success == 0, torch.where((grasp_reward > grasp_success_dist), torch.ones_like(grasp_success), grasp_success), grasp_success)
    lift_success = torch.where(lift_success == 0, torch.where((lift_reward > lift_success_dist), torch.ones_like(lift_success), lift_success), lift_success)
    
    cons_successes = torch.where(   num_resets > 0, 
                                    av_factor * finished_cons_successes / num_resets + 
                                    (1.0 - av_factor) * consecutive_successes, 
                                    consecutive_successes   )

    return reward, resets, reset_goal_buf, progress_buf, grasp_success, lift_success, cons_successes
