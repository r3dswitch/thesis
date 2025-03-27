import numpy as np
import os
import torch
import math

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, tensor_clamp, \
    tf_vector, tf_combine, quat_conjugate, quat_mul
from .base.vec_task import VecTask

def iprint(*strings):
    print(strings)
    exit()

class ArnieGrasp(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.num_props = self.cfg["env"]["numProps"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self.cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self.cfg["env"]["openRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2
        self.distX_offset = 0.04
        self.dt = 1/60.

        num_obs = 59
        num_acts = 27

        self.cfg["env"]["numObservations"] = 57
        self.cfg["env"]["numActions"] = 27

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.diana_default_dof_pos = to_torch([-0.9,-1.4,-3.1,0.8,3.1,0.5,-0.4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,], device=self.device)
        
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        
        self.diana_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_diana_dofs]
        self.diana_dof_pos = self.diana_dof_state[..., 0]
        self.diana_dof_vel = self.diana_dof_state[..., 1]
        
        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_diana_dofs:]
        self.object_dof_pos = self.object_dof_state[..., 0]
        self.object_dof_vel = self.object_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.diana_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * (2 + self.num_props), dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
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

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        diana_asset_file = "urdf/right_hit.urdf"
        table_asset_file = "urdf/square_table.urdf"
        object_asset_file = "urdf/cube.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        
        diana_asset = self.gym.load_asset(self.sim, asset_root, diana_asset_file, asset_options)
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)
        table_asset = self.gym.load_asset(self.sim, asset_root, table_asset_file, asset_options)

        self.num_diana_bodies = self.gym.get_asset_rigid_body_count(diana_asset)
        self.num_diana_dofs = self.gym.get_asset_dof_count(diana_asset)
        diana_dof_props = self.gym.get_asset_dof_properties(diana_asset)
        
        self.diana_dof_lower_limits = []
        self.diana_dof_upper_limits = []
        
        for i in range(self.num_diana_dofs):
            diana_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            diana_dof_props['stiffness'][i] = 7000.0
            diana_dof_props['damping'][i] = 50.0

            self.diana_dof_lower_limits.append(diana_dof_props['lower'][i])
            self.diana_dof_upper_limits.append(diana_dof_props['upper'][i])

        self.diana_dof_lower_limits = to_torch(self.diana_dof_lower_limits, device=self.device)
        self.diana_dof_upper_limits = to_torch(self.diana_dof_upper_limits, device=self.device)
        self.diana_dof_speed_scales = torch.ones_like(self.diana_dof_lower_limits)
        self.diana_dof_speed_scales[[7, 8]] = 0.1
        
        diana_dof_props['effort'][7] = 200
        diana_dof_props['effort'][8] = 200

        diana_start_pose = gymapi.Transform()
        diana_start_pose.p = gymapi.Vec3(1, -0.1, 1.45)
        diana_start_pose.r = gymapi.Quat().from_euler_zyx(1.5652925671162337, 0, 0.227)

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0.4,0,0.4)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.4,0,1.1)

        self.dianas = []
        self.objects = []
        self.tables = []
        self.envs = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            diana_actor = self.gym.create_actor(env_ptr, diana_asset, diana_start_pose, "diana", i, 1, 0)
            self.gym.set_actor_dof_properties(env_ptr, diana_actor, diana_dof_props)

            object_actor = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "cube", i, 1, 0)
            
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 2, 0)

            self.envs.append(env_ptr)
            self.dianas.append(diana_actor)
            self.objects.append(object_actor)

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "diana")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        self.hand_index = self.gym.get_asset_rigid_body_dict(diana_asset)["link_7"]
        self.j_eef = self.jacobian[:, self.hand_index - 1, :, :7]

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, diana_actor, "link_7")
        self.object_handle = self.gym.find_actor_rigid_body_handle(env_ptr, object_actor, "object")
        
        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.dianas[0], "link_7")

        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        hand_pos = to_torch([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        hand_rot = to_torch([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.hand_pos = self.object_pos = torch.zeros_like(hand_pos)
        self.hand_rot = self.object_rot = torch.zeros_like(hand_rot)

        hand_pose_inv = hand_pose.inverse()
        grasp_pose_axis = 1
        
        diana_local_grasp_pose = hand_pose_inv
        diana_local_grasp_pose.p += gymapi.Vec3(*get_axis_params(0.04, grasp_pose_axis))
        
        self.diana_local_grasp_pos = to_torch([diana_local_grasp_pose.p.x, diana_local_grasp_pose.p.y,
                                                diana_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.diana_local_grasp_rot = to_torch([diana_local_grasp_pose.r.x, diana_local_grasp_pose.r.y,
                                                diana_local_grasp_pose.r.z, diana_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        object_local_grasp_pose = gymapi.Transform()
        object_local_grasp_pose.p = gymapi.Vec3(*get_axis_params(0.01, grasp_pose_axis, 0.3))
        object_local_grasp_pose.r = gymapi.Quat(0, 0, 0, 1)
        
        self.object_local_grasp_pos = to_torch([object_local_grasp_pose.p.x, object_local_grasp_pose.p.y,
                                                object_local_grasp_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.object_local_grasp_rot = to_torch([object_local_grasp_pose.r.x, object_local_grasp_pose.r.y,
                                                object_local_grasp_pose.r.z, object_local_grasp_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.diana_grasp_pos = torch.zeros_like(self.diana_local_grasp_pos)
        self.diana_grasp_rot = torch.zeros_like(self.diana_local_grasp_rot)
        self.diana_grasp_rot[..., -1] = 1  
        
        self.object_grasp_pos = torch.zeros_like(self.object_local_grasp_pos)
        self.object_grasp_rot = torch.zeros_like(self.object_local_grasp_rot)
        self.object_grasp_rot[..., -1] = 1

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:] = compute_diana_reward(
            self.reset_buf, self.progress_buf, self.actions,
            self.diana_grasp_pos, self.object_grasp_pos, self.diana_grasp_rot, self.object_grasp_rot,
            self.num_envs, self.dist_reward_scale,
            self.action_penalty_scale, self.distX_offset, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]
       
        self.object_pos = self.rigid_body_states[:, self.object_handle][:, 0:3]
        self.object_rot = self.rigid_body_states[:, self.object_handle][:, 3:7]

        self.diana_grasp_rot[:], self.diana_grasp_pos[:], self.object_grasp_rot[:], self.object_grasp_pos[:] = \
            compute_grasp_transforms(   self.hand_rot, self.hand_pos, 
                                        self.diana_local_grasp_rot, self.diana_local_grasp_pos,
                                        self.object_rot, self.object_pos, 
                                        self.object_local_grasp_rot, self.object_local_grasp_pos
                                     )

        dof_pos_scaled = (2.0 * (self.diana_dof_pos - self.diana_dof_lower_limits)
                          / (self.diana_dof_upper_limits - self.diana_dof_lower_limits) - 1.0)
        
        to_target = self.object_grasp_pos - self.diana_grasp_pos
        
        self.obs_buf = torch.cat((dof_pos_scaled, self.diana_dof_vel * self.dof_vel_scale, to_target), dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):

        pos = tensor_clamp( self.diana_default_dof_pos.unsqueeze(0) + 0.25 * (torch.rand((len(env_ids), self.num_diana_dofs), device=self.device) - 0.5), 
                            self.diana_dof_lower_limits, 
                            self.diana_dof_upper_limits )

        self.diana_dof_pos[env_ids, :] = pos
        self.diana_dof_vel[env_ids, :] = torch.zeros_like(self.diana_dof_vel[env_ids])
        self.diana_dof_targets[env_ids, :self.num_diana_dofs] = pos

        self.object_dof_state[env_ids, :] = torch.zeros_like(self.object_dof_state[env_ids])

        multi_env_ids_int32 = self.global_indices[env_ids, :2].flatten()
        
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.diana_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        use_ik = True
        if use_ik:
            self.target_pos = self.object_pos.clone()
            
            target_rot = gymapi.Quat.from_euler_zyx(0, 0, 0)  
            # self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)
            self.target_rot = torch.tensor([[-0.467, -0.169, -0.077, -0.865]] * self.num_envs, dtype=torch.float32).to(self.device)

            if self.progress_buf[0] > 0 and self.progress_buf[0] < 60:
                pos_err = self.target_pos - self.hand_pos
            elif self.progress_buf[0] > 59 and self.progress_buf[0] < 90:
                self.target_pos[:,2] -= 0.25
                pos_err = self.target_pos - self.hand_pos
            elif self.progress_buf[0] > 89:
                # self.target_pos[:,2] -= 0.25
                pos_err = self.target_pos - self.hand_pos
            else:
                pos_err = self.target_pos - self.hand_pos

            orn_err = orientation_error(self.target_rot, self.hand_rot)

            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1) 
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            damping = 0.05
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 7)
            
            targets = self.diana_dof_targets[:, :self.num_diana_dofs]
            targets[:, :7] = self.diana_dof_pos[:, :7] + u.squeeze(-1)
            self.diana_dof_targets[:, :self.num_diana_dofs] = tensor_clamp(targets, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
        
        else: 
            targets = self.diana_dof_targets[:, :self.num_diana_dofs] + self.diana_dof_speed_scales * self.dt * self.actions * self.action_scale
            self.diana_dof_targets[:, :self.num_diana_dofs] = tensor_clamp(targets, self.diana_dof_lower_limits, self.diana_dof_upper_limits)
            env_ids_int32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.diana_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        
        self.compute_observations()
        self.compute_reward(self.actions)

@torch.jit.script
def compute_diana_reward(
    reset_buf, progress_buf, actions,
    diana_grasp_pos, object_pos, diana_grasp_rot, object_rot,
    num_envs, dist_reward_scale,
    action_penalty_scale, distX_offset, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float, float, float) -> Tuple[Tensor, Tensor]

    d = torch.norm(diana_grasp_pos - object_pos, p=2, dim=-1)
    dist_reward = 1.0 / (1.0 + d ** 2)
    dist_reward *= dist_reward
    dist_reward = torch.where(d <= 0.02, dist_reward * 2, dist_reward)

    action_penalty = torch.sum(actions ** 2, dim=-1)

    rewards = dist_reward_scale * dist_reward - action_penalty_scale * action_penalty
    
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf

@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, diana_local_grasp_rot, diana_local_grasp_pos,
                             object_rot, object_pos, object_local_grasp_rot, object_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_diana_rot, global_diana_pos = tf_combine(hand_rot, hand_pos, diana_local_grasp_rot, diana_local_grasp_pos)
    global_object_rot, global_object_pos = tf_combine(object_rot, object_pos, object_local_grasp_rot, object_local_grasp_pos)

    return global_diana_rot, global_diana_pos, global_object_rot, global_object_pos

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)