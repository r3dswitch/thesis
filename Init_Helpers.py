from isaacgym import gymapi, gymtorch
import torch, glob, os
from isaacgym.torch_utils import *

def initialize_config(cfg, device_type, device_id, headless):
    """Initialize configuration parameters."""
    my_cfg = cfg
    my_cfg["env"]["numActions"] = 27
    my_cfg["device_type"] = device_type
    my_cfg["device_id"] = device_id
    my_cfg["headless"] = headless
    return my_cfg

def initialize_task_params(obj, cfg):
    obj.randomize = cfg["task"]["randomize"]
    obj.randomization_params = cfg["task"]["randomization_params"]
    obj.aggregate_mode = cfg["env"]["aggregateMode"]
    
    obj.dist_reward_scale = cfg["env"]["distRewardScale"]
    obj.rot_reward_scale = cfg["env"]["rotRewardScale"]
    obj.action_penalty_scale = cfg["env"]["actionPenaltyScale"]
    
    obj.success_tolerance = cfg["env"]["successTolerance"]
    obj.reach_goal_bonus = cfg["env"]["reachGoalBonus"]
    obj.fall_dist = cfg["env"]["fallDistance"]
    obj.fall_penalty = cfg["env"]["fallPenalty"]
    obj.rot_eps = cfg["env"]["rotEps"]
    
    obj.reset_position_noise = cfg["env"]["resetPositionNoise"]
    obj.reset_rotation_noise = cfg["env"]["resetRotationNoise"]
    obj.reset_dof_pos_noise = cfg["env"]["resetDofPosRandomInterval"]
    obj.reset_dof_vel_noise = cfg["env"]["resetDofVelRandomInterval"]
    
    obj.max_episode_length = cfg["env"]["maxEpisodeLength"]
    obj.print_success_stat = cfg["env"]["printNumSuccesses"]
    obj.max_consecutive_successes = cfg["env"]["maxConsecutiveSuccesses"]
    obj.av_factor = cfg["env"].get("averFactor", 0.1)
    obj.act_avg = cfg["env"]["actionsMovingAverage"]
    
    obj.up_axis = cfg["env"]["upAxis"]
    obj.debug = cfg["env"]["debug"]
    
    # obj.knock_dist = cfg["env"]["knockDist"]
    # obj.ik_fail_dist = cfg["env"]["failDistIK"]
    # obj.fail_dist = cfg["env"]["failDistHand"]
    # obj.grasp_success_dist = cfg["env"]["graspSuccessDist"]
    # obj.lift_success_dist = cfg["env"]["liftSuccessDist"]

    return obj, cfg

def setup_camera(obj, cam_pos = gymapi.Vec3(1, 0, 1.5), cam_target = gymapi.Vec3(0.2, 0.2, 0.75)):
    if obj.viewer is not None:
        obj.gym.viewer_camera_look_at(obj.viewer, None, cam_pos, cam_target)

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
    
    obj.target_euler = to_torch([-1.57, 3.0, 0.0], device=obj.device).repeat((obj.num_envs, 1))
    obj.cup_pos = to_torch([0.2, 0, 1], device=obj.device).repeat((obj.num_envs, 1))

def setup_hand_positions(obj):
    init_pos = obj.cfg["env"]["handPositions"]["defaultPos"]
    traj_list = obj.cfg["env"]["handPositions"]["trajectoryPosList"]
    
    obj.hand_dof_default_pos = torch.zeros(
        obj.num_hand_dofs, dtype=torch.float, device=obj.device
    )
    obj.hand_dof_default_pos[:7] = torch.tensor( init_pos, dtype=torch.float, device=obj.device )

    obj.trajectory_list = []
    
    for i in range(len(traj_list)):
        obj.trajectory = to_torch(traj_list[i], dtype=torch.float, device=obj.device)
        obj.trajectory_list.append(obj.trajectory)
    
def setup_dataset(obj):
    obj.dataset_basedir = obj.cfg["env"]["graspDataset"]["baseDir"]
    obj.mesh_folder = obj.cfg["env"]["graspDataset"]["meshDir"]
    is_full_dataset = obj.cfg["env"]["graspDataset"]["isFullDataset"]
    grasp_path = obj.cfg["env"]["graspDataset"]["graspPath"]
    dataset_regex = obj.cfg["env"]["graspDataset"]["datasetRegex"]
    
    obj.pt_dataset_path = glob.glob(os.path.join(obj.dataset_basedir, dataset_regex))
    
    obj.grasp_list = []
    
    if is_full_dataset:
        for dataset_path_i in obj.pt_dataset_path:
            graspdata = torch.load(dataset_path_i)["metadata"]
            for index_candidate in range(obj.num_envs):
                hand_pose = graspdata[index_candidate][1]           # hand pose
                obj.grasp_list.append(hand_pose)

        obj.grasp_list = torch.stack(obj.grasp_list).to(obj.device)
    else:
        grasps = np.load(grasp_path)
        grasps = np.reshape(grasps[0], (1,29)) #######
        obj.grasp_list = torch.from_numpy(grasps).to(obj.device)
        if obj.num_envs < obj.grasp_list.shape[0]:
            obj.grasp_list = obj.grasp_list[:obj.num_envs,:]
        else:
            stack_num = (int) (obj.num_envs / obj.grasp_list.shape[0]) + 1
            obj.grasp_list = obj.grasp_list.repeat(stack_num, 1)

def setup_stages(obj):
    obj.prempt_start = obj.cfg["env"]["stages"]["preempt"]
    obj.ik_start = obj.cfg["env"]["stages"]["ik"]
    obj.grasp_start = obj.cfg["env"]["stages"]["grasp"]
    obj.rl_start = obj.cfg["env"]["stages"]["rl"]
    obj.lift_start = obj.cfg["env"]["stages"]["lift"]
