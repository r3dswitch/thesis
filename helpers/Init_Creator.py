from isaacgym import gymapi, gymtorch
import torch, glob, os
from isaacgym.torch_utils import *


class ConfigFactory:
    @staticmethod
    def initialize_config(cfg, device_type, device_id, headless):
        """Initialize and return configuration parameters."""
        my_cfg = cfg
        my_cfg["env"]["numActions"] = 27
        my_cfg["device_type"] = device_type
        my_cfg["device_id"] = device_id
        my_cfg["headless"] = headless
        return my_cfg

    @staticmethod
    def initialize_task_params(obj, cfg):
        """Initialize task parameters for an object based on configuration."""
        task_params = {
            'randomize': cfg["task"]["randomize"],
            'randomization_params': cfg["task"]["randomization_params"],
            'aggregate_mode': cfg["env"]["aggregateMode"],
            'dist_reward_scale': cfg["env"]["distRewardScale"],
            'rot_reward_scale': cfg["env"]["rotRewardScale"],
            'action_penalty_scale': cfg["env"]["actionPenaltyScale"],
            'success_tolerance': cfg["env"]["successTolerance"],
            'reach_goal_bonus': cfg["env"]["reachGoalBonus"],
            'fall_dist': cfg["env"]["fallDistance"],
            'fall_penalty': cfg["env"]["fallPenalty"],
            'rot_eps': cfg["env"]["rotEps"],
            'reset_position_noise': cfg["env"]["resetPositionNoise"],
            'reset_rotation_noise': cfg["env"]["resetRotationNoise"],
            'reset_dof_pos_noise': cfg["env"]["resetDofPosRandomInterval"],
            'reset_dof_vel_noise': cfg["env"]["resetDofVelRandomInterval"],
            'max_episode_length': cfg["env"]["maxEpisodeLength"],
            'print_success_stat': cfg["env"]["printNumSuccesses"],
            'max_consecutive_successes': cfg["env"]["maxConsecutiveSuccesses"],
            'av_factor': cfg["env"].get("averFactor", 0.1),
            'act_avg': cfg["env"]["actionsMovingAverage"],
            'up_axis': cfg["env"]["upAxis"],
            'debug': cfg["env"]["debug"],
            # 'knock_dist': cfg["env"]["knockDist"],
            # 'ik_fail_dist': cfg["env"]["failDistIK"],
            # 'fail_dist': cfg["env"]["failDistHand"],
            # 'grasp_success_dist': cfg["env"]["graspSuccessDist"],
            # 'lift_success_dist': cfg["env"]["liftSuccessDist"],
            'num_obs': cfg["env"]["numObservations"]
        }

        for param, value in task_params.items():
            setattr(obj, param, value)
        
        return obj, cfg

    @staticmethod
    def setup_camera(obj, cam_pos=gymapi.Vec3(1, 0, 1.5), cam_target=gymapi.Vec3(0.2, 0.2, 0.75)):
        """Setup camera for the simulation environment."""
        if obj.viewer is not None:
            obj.gym.viewer_camera_look_at(obj.viewer, None, cam_pos, cam_target)

    @staticmethod
    def setup_tensors(obj):
        obj.actor_root_state_tensor = obj.gym.acquire_actor_root_state_tensor(obj.sim)
        obj.dof_state_tensor = obj.gym.acquire_dof_state_tensor(obj.sim)
        obj.rigid_body_tensor = obj.gym.acquire_rigid_body_state_tensor(obj.sim)
        obj.jacobian_tensor = gymtorch.wrap_tensor(obj.gym.acquire_jacobian_tensor(obj.sim, obj.cfg["env"]["jacobianActorName"]))

        obj.gym.refresh_actor_root_state_tensor(obj.sim)
        obj.gym.refresh_dof_state_tensor(obj.sim)
        obj.gym.refresh_rigid_body_state_tensor(obj.sim)
        
        obj.dof_state = gymtorch.wrap_tensor(obj.dof_state_tensor)
        obj.hand_dof_state = obj.dof_state.view(obj.num_envs, -1, 2)[:, :obj.num_hand_dofs]
        obj.hand_dof_pos = obj.hand_dof_state[..., 0]
        obj.hand_dof_vel = obj.hand_dof_state[..., 1]

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
        
        obj.object_pos = obj.root_state_tensor[obj.object_indices, 0:3].clone()
        obj.object_rot = obj.object_init_rot = obj.root_state_tensor[obj.object_indices, 3:7].clone()
        obj.object_init_pos = obj.grasp_pos = obj.intermediate_pos = obj.handle_pos = obj.object_pos
        obj.object_init_rot = obj.grasp_rot = obj.intermediate_rot = obj.handle_rot = obj.object_rot
        obj.wrist_pos = obj.root_state_tensor[obj.hand_indices, 0:3].clone()
        obj.wrist_rot = obj.root_state_tensor[obj.hand_indices, 3:7].clone()
        
        obj.wrist_rigid_body_index = obj.gym.find_actor_rigid_body_index(obj.envs[0], obj.hand_indices[0], obj.cfg["env"]["endEffectorLinkName"], gymapi.DOMAIN_ENV)
        obj.wrist_rot_mod = torch.tensor(obj.cfg["env"]["wristRotModQuaternion"],device=obj.device).repeat(obj.num_envs, 1)
        
        obj.reset_goal_buf = obj.reset_buf.clone()
        obj.grasp_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.lift_success = torch.zeros(obj.num_envs, dtype=torch.float, device=obj.device)
        obj.consecutive_successes = torch.zeros(1, dtype=torch.float, device=obj.device)
        
        obj.target_euler = to_torch([0.0, -1.57, 0.0], device=obj.device).repeat((obj.num_envs, 1))

    @staticmethod
    def setup_hand_positions(obj):
        """Initialize hand positions and trajectories."""
        init_pos = obj.cfg["env"]["handPositions"]["defaultPos"]
        traj_list = obj.cfg["env"]["handPositions"]["trajectoryPosList"]
        
        obj.hand_dof_default_pos = torch.zeros(obj.num_hand_dofs, dtype=torch.float, device=obj.device)
        obj.hand_dof_default_pos[:7] = torch.tensor(init_pos, dtype=torch.float, device=obj.device)

        obj.trajectory_list = [to_torch(pos, dtype=torch.float, device=obj.device) for pos in traj_list]

    @staticmethod
    def setup_dataset(obj):
        """Setup grasp dataset for the simulation environment."""
        obj.dataset_basedir = obj.cfg["env"]["graspDataset"]["baseDir"]
        obj.mesh_folder = obj.cfg["env"]["graspDataset"]["meshDir"]
        is_full_dataset = obj.cfg["env"]["graspDataset"]["isFullDataset"]
        grasp_path = obj.cfg["env"]["graspDataset"]["graspPath"]
        dataset_regex = obj.cfg["env"]["graspDataset"]["datasetRegex"]

        obj.pt_dataset_path = glob.glob(os.path.join(obj.dataset_basedir, dataset_regex))
        obj.grasp_list = []

        if is_full_dataset:
            for dataset_path in obj.pt_dataset_path:
                graspdata = torch.load(dataset_path)["metadata"]
                for index_candidate in range(obj.num_envs):
                    hand_pose = graspdata[index_candidate][1]
                    obj.grasp_list.append(hand_pose)

            obj.grasp_list = torch.stack(obj.grasp_list).to(obj.device)
        else:
            grasps = np.load(grasp_path)
            obj.grasp_list = torch.from_numpy(np.reshape(grasps[0], (1, 29))).to(obj.device)
            if obj.num_envs < obj.grasp_list.shape[0]:
                obj.grasp_list = obj.grasp_list[:obj.num_envs, :]
            else:
                obj.grasp_list = obj.grasp_list.repeat((obj.num_envs // obj.grasp_list.shape[0]) + 1, 1)

    @staticmethod
    def setup_stages(obj):
        """Initialize stages based on configuration."""
        stage_params = {
            'prempt_start': obj.cfg["env"]["stages"]["preempt"],
            'ik_start': obj.cfg["env"]["stages"]["ik"],
            'grasp_start': obj.cfg["env"]["stages"]["grasp"],
            'rl_start': obj.cfg["env"]["stages"]["rl"],
            'lift_start': obj.cfg["env"]["stages"]["lift"]
        }

        for param, value in stage_params.items():
            setattr(obj, param, value)
