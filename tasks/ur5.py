# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np
import os
from numpy.core.numeric import zeros_like
from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import to_torch, tf_combine, quat_apply, quat_conjugate, quat_mul, tensor_clamp
from tasks.base.vec_task import VecTask
import torch
from IPython import embed       ### using for debug
import math
import matplotlib.pyplot as plt
from PIL import Image as Im
import torch.nn as nn
from einops.einops import rearrange
import random
# from tasks.base.experiment import VAEXperiment
# from tasks.base.base import BaseVAE
from tasks.base.beta_vae import BetaVAE
from tasks.base.vanilla_vae import VanillaVAE
from torchvision import transforms
import rl_games
from cyclegan.fake_image import get_fake_img

class UR5PushAndGrasp(VecTask):
    def __init__(self, cfg, sim_device, graphics_device_id, headless):
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
        self.finger_dist_reward_scale = self.cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["200x200x60", "230x230x60", "250x200x60", "250x230x60", "250x250x60","random"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        self.distX_offset = 0.04
        self.dt = 1/60.

        num_obs = 135 # 37635 # 15
        num_acts = 2
        self.scale = 1.5

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        # Camera Sensor
        self.camera_props = gymapi.CameraProperties()
        self.camera_props.width = 960
        self.camera_props.height = 540
        self.camera_props.horizontal_fov = 84.1   
        self.camera_props.enable_tensors = True
        self.debug_fig = plt.figure("debug")

        super().__init__(config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless)
       
        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.ur5_default_dof_pos = to_torch([0.8554, -0.5559, 1.1894, -0.6763, -1.5823, -0.6306, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=self.device)#
        
        # Dof state slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.ur5_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_ur5_dofs]
        self.ur5_dof_pos = self.ur5_dof_state[..., 0]
        self.ur5_dof_vel = self.ur5_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)  # 3p+4r+3v+3w
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(self.num_envs, -1, 13)
        self.box_states = self.root_state_tensor[:, 2:]  # mention the range of box_states 
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.ur5_dof_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)  ###########

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1) ###############this place 2 depend on how many obj in env
        # self.reset_idx(torch.arange(self.num_envs, device=self.device))

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

        asset_root = "assets"
        ur5_asset_file = "tams_ur5_urdf/tams_robot.urdf"
        table_asset_file = "tams_ur5_urdf/tams_corner.urdf"
        object_asset_file = self.asset_files_dict[self.object_type]

        # load ur5 asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        ur5_asset = self.gym.load_asset(self.sim, asset_root, ur5_asset_file, asset_options)

        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.thickness = 0.001
        asset_options.use_mesh_materials = True
        asset_options.density = 400
        asset_options.fix_base_link = True
        table_asset = self.gym.load_asset(self.sim, asset_root, table_asset_file, asset_options)

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.use_mesh_materials = True
        asset_options.density = 400
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)


        ur5_dof_stiffness = to_torch([400, 400, 400, 400, 400, 400, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4, 1.0e4], dtype=torch.float, device=self.device)
        ur5_dof_damping = to_torch([80, 80, 80, 80, 80, 80, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2, 1.0e2], dtype=torch.float, device=self.device)
      
        self.num_ur5_bodies = self.gym.get_asset_rigid_body_count(ur5_asset)
        self.num_ur5_dofs = self.gym.get_asset_dof_count(ur5_asset)
        self.num_table_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        self.num_table_dofs = self.gym.get_asset_dof_count(table_asset) 

        ur5_dof_props = self.gym.get_asset_dof_properties(ur5_asset)
        self.ur5_dof_lower_limits = []
        self.ur5_dof_upper_limits = []
        for i in range(self.num_ur5_dofs):
            ur5_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur5_dof_props['stiffness'][i] = ur5_dof_stiffness[i]
                ur5_dof_props['damping'][i] = ur5_dof_damping[i]
            else:
                ur5_dof_props['stiffness'][i] = 7000.0
                ur5_dof_props['damping'][i] = 50.0

            self.ur5_dof_lower_limits.append(ur5_dof_props['lower'][i])
            self.ur5_dof_upper_limits.append(ur5_dof_props['upper'][i])

        self.ur5_dof_lower_limits = to_torch(self.ur5_dof_lower_limits, device=self.device)
        self.ur5_dof_upper_limits = to_torch(self.ur5_dof_upper_limits, device=self.device)
        self.ur5_dof_speed_scales = torch.ones_like(self.ur5_dof_lower_limits)

        table_dof_props = self.gym.get_asset_dof_properties(table_asset)
        for i in range(self.num_table_dofs):
            table_dof_props['damping'][i] = 10.0


        ur5_start_pose = gymapi.Transform()
        ur5_start_pose.p = gymapi.Vec3(0.226, 0.7, 1.085)  # do bot let hand interface with table
        ur5_start_pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)  ###############

        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(0, 0, 0)  # do bot let hand interface with table

        box_pose = gymapi.Transform()

        num_ur5_bodies = self.gym.get_asset_rigid_body_count(ur5_asset)
        num_ur5_shapes = self.gym.get_asset_rigid_shape_count(ur5_asset)
        num_tabel_bodies = self.gym.get_asset_rigid_body_count(table_asset)
        num_table_shapes = self.gym.get_asset_rigid_shape_count(table_asset)
        num_box_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        num_box_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
        max_agg_bodies = num_ur5_bodies + num_tabel_bodies + num_box_bodies
        max_agg_shapes = num_ur5_shapes + num_table_shapes + num_box_shapes

        self.ur5s = []
        self.bases = []
        self.default_prop_states = []
        self.prop_start = []
        self.envs = []
        self.camera_handles = []
        self.shafts = []
        self.tables =  []
        self.shaft_poses = []
        self.lfinger_poses = []
        self.rfinger_poses = []
        self.boxs = []
        self.default_box_states = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
            transform = gymapi.Transform()
            transform.p = gymapi.Vec3(1.4403, 0.94314, 1.3286)    # (1., 1.05, 1.1)
            transform.r = gymapi.Quat(-0.20037099181882714 , 0.014480071675084041 , 0.9794848483640505 , 0.015850078456497597) # y -90 -> x 90
            self.gym.set_camera_transform(camera_handle, env_ptr, transform)

            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            ur5_actor = self.gym.create_actor(env_ptr, ur5_asset, ur5_start_pose, "ur5", i, 1)   # here 1 is more accurate  
            
            self.gym.set_actor_dof_properties(env_ptr, ur5_actor, ur5_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i)   # here 1 is more accurate  
            self.gym.set_actor_dof_properties(env_ptr, table_actor, table_dof_props)

            object_asset_file = self.cfg["env"]["asset"].get(i % 80, self.asset_files_dict[i%80])
            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.use_mesh_materials = True
            asset_options.density = 400
            object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, asset_options)

            box_pose.p.x = 0.7 + np.random.uniform(-0.125, 0.125)    # 0.7
            box_pose.p.y = 1.05 + np.random.uniform(-0.1, 0.1)  # 0.8
            box_pose.p.z = 0.78 #  0.78 + 0.5 * box_size/4
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))  #########  # roat an angle around z-axis
            box_handle = self.gym.create_actor(env_ptr, object_asset, box_pose, "box", i)
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

            self.default_box_states.append([box_pose.p.x, box_pose.p.y, box_pose.p.z,
                                                         box_pose.r.x, box_pose.r.y, box_pose.r.z, box_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.ur5s.append(ur5_actor)
            self.tables.append(table_actor)
            self.boxs.append(box_handle)
            self.camera_handles.append(camera_handle)

        self.hand_handle = self.gym.find_actor_rigid_body_handle(env_ptr, ur5_actor, "ur5_wrist_3_link")
        self.box_handle = self.gym.find_actor_rigid_body_handle(env_ptr, box_handle, "box")

        self.default_box_states = to_torch(self.default_box_states, device=self.device, dtype=torch.float).view(self.num_envs, -1 , 13)

        self._jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5")
        self.jacobian = gymtorch.wrap_tensor(self._jacobian)
        self.hand_index = self.gym.get_asset_rigid_body_dict(ur5_asset)["ur5_wrist_3_link"]
        self.j_eef = self.jacobian[:, self.hand_index - 1, :, :6]   #17
        self.init_data()

    def init_data(self):
        hand = self.gym.find_actor_rigid_body_handle(self.envs[0], self.ur5s[0], "ur5_wrist_3_link")
        
        hand_pose = self.gym.get_rigid_transform(self.envs[0], hand)
        
        self.ur5_hand_pos = to_torch([hand_pose.p.x, hand_pose.p.y,
                                                hand_pose.p.z], device=self.device).repeat((self.num_envs, 1))
        self.ur5_hand_rot = to_torch([hand_pose.r.x, hand_pose.r.y,
                                                hand_pose.r.z, hand_pose.r.w], device=self.device).repeat((self.num_envs, 1))

        self.hand_pos = torch.zeros_like(self.ur5_hand_pos)
        self.hand_rot = torch.zeros_like(self.ur5_hand_rot)

    def compute_reward(self, actions):
        self.rew_buf[:], self.reset_buf[:], self.success_buf[:] = compute_ur5_reward(
            self.reset_buf, self.progress_buf, self.actions, self.box_pos, self.hand_pos,
            self.num_envs, self.dist_reward_scale, self.rot_reward_scale, self.around_handle_reward_scale, self.open_reward_scale,
            self.finger_dist_reward_scale, self.action_penalty_scale, self.scale, self.max_episode_length
        )

    def compute_observations(self):
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)

        self.hand_pos = self.rigid_body_states[:, self.hand_handle][:, 0:3]
        self.hand_rot = self.rigid_body_states[:, self.hand_handle][:, 3:7]

        self.box_pos = self.rigid_body_states[:, self.box_handle][:, 0:3]
        self.box_rot = self.rigid_body_states[:, self.box_handle][:, 3:7]
        
        # to_target = self.box_pos[:, 1] - 0.9

        # visual input
        camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR)
        torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
        torch_camera_tensor = to_torch(torch_camera_tensor, dtype=torch.float, device=self.device).unsqueeze(0)
        self.img_buf = torch_camera_tensor
        # self.img_buf = self.img_buf[:, 180:540, 310:720, :3]

        for i in range(1, self.num_envs):
            camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[i], self.camera_handles[i], gymapi.IMAGE_COLOR)
            torch_camera_tensor = gymtorch.wrap_tensor(camera_tensor)
            torch_camera_tensor = to_torch(torch_camera_tensor, dtype=torch.float, device=self.device).unsqueeze(0)
            self.img_buf = torch.cat((self.img_buf, torch_camera_tensor), dim=0)

        # self.img_buf = self.img_buf[:, 37:261, 37:261, :3]
        self.img_buf = self.img_buf[:, 180:540, 310:720, :3]
        self.img_buf = rearrange(self.img_buf, 'b h w c-> b c h w' )

        torch_resize = transforms.Resize([80,64])
        self.img_buf = torch_resize(self.img_buf)

        self.img_buf = self.img_buf[:, :, :, :]/255
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        for c in range(3):
            self.img_buf[:, c, :, :] = (self.img_buf[:, c, :, :] - image_mean[c])/image_std[c]
        # self.img_buf = rearrange(self.img_buf, 'b h w c -> b c h w')
        
        model = VanillaVAE(in_channels = 3,latent_dim = 128 )
        model.load_state_dict(torch.load(os.path.abspath("model/vae_179.pth")))
        model.cuda()

        with torch.no_grad():
            object_recon, input, object_mu, log_var = model(self.img_buf)
        rgb_obs = object_mu

        self.obs_buf = torch.cat((self.hand_pos, self.hand_rot, rgb_obs), dim=1)
        return self.obs_buf

    def reset_idx(self, env_ids):
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        l_color = gymapi.Vec3(random.uniform(1, 1), random.uniform(1, 1), random.uniform(1, 1))
        l_ambient = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        l_direction = gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        self.gym.set_light_parameters(self.sim, 0, l_color, l_ambient, l_direction)

        pos = tensor_clamp(
            self.ur5_default_dof_pos.unsqueeze(0),
            self.ur5_dof_lower_limits, self.ur5_dof_upper_limits)
        self.ur5_dof_pos[env_ids, :] = pos
        self.ur5_dof_vel[env_ids, :] = torch.zeros_like(self.ur5_dof_vel[env_ids])
        self.ur5_dof_targets[env_ids, :self.num_ur5_dofs] = pos
        box_indices = self.global_indices[env_ids, 2].flatten()

        self.reset_box_states = []
        for i in range(self.num_envs):
            envPtr = self.envs[i]
            object = self.boxs[i]
            box_pose = gymapi.Transform()
            box_pose.p.x = 0.7 + np.random.uniform(-0.125, 0.125)       ##################
            box_pose.p.y = 1.05 + np.random.uniform(-0.1, 0.1)  # 0.8
            box_pose.p.z = 0.78 # 0.78 + 0.5 * 0.25/4
            color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
            self.gym.set_rigid_body_color(envPtr, object, 0, gymapi.MESH_VISUAL, color)
            box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
            self.reset_box_states.append([box_pose.p.x, box_pose.p.y, box_pose.p.z,
                                                         box_pose.r.x, box_pose.r.y, box_pose.r.z, box_pose.r.w,
                                                         0, 0, 0, 0, 0, 0])
        self.reset_box_states = to_torch(self.reset_box_states, device=self.device, dtype=torch.float).view(self.num_envs, -1 , 13)
        self.root_state_tensor[env_ids, 2] = self.reset_box_states[env_ids].squeeze(1)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(box_indices), len(box_indices))

        multi_env_ids_int32 = self.global_indices[env_ids, 0].flatten()  # , :1
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.ur5_dof_targets),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32), len(multi_env_ids_int32))
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
    def pre_physics_step(self, actions):   # actions in range [-1,1] 
        i = 0
        self.actions = actions.clone().to(self.device)
    
        if self.progress_buf[i] <= 144:
            self.target_pos = self.hand_pos.clone()
            self.target_pos[:,0] += self.actions[:,0] *self.dt * self.action_scale
            self.target_pos[:,1] += self.actions[:,1] *self.dt * self.action_scale
            self.target_pos[:,2] = 1.17
            limits = torch.ones_like(self.actions[:,1])
            self.target_pos[:,0] = torch.where(self.target_pos[:,0] > 0.9, limits*0.9, self.target_pos[:,0])
            self.target_pos[:,0] = torch.where(self.target_pos[:,0] < 0.4, limits*0.4, self.target_pos[:,0])
            self.target_pos[:,1] = torch.where(self.target_pos[:,1] > 1.35, limits*1.35, self.target_pos[:,1])
        
            target_rot = gymapi.Quat.from_euler_zyx(-0.5*math.pi, 0, 0)  # x,y,z
            self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)

            pos_err = self.target_pos - self.hand_pos
            orn_err = orientation_error(self.target_rot, self.hand_rot)

            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # unsqueeze(-1) meaning up one dim internal
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            damping = 0.05
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)      ###########17
            self.ur5_dof_targets[:, :6] = self.ur5_dof_pos[:, :6] + u.squeeze(-1)       # 17
    
        if self.progress_buf[i] > 144 and self.progress_buf[i] <= 147:
            self.target_pos = self.hand_pos.clone()
            self.target_pos[:,1] += 0.2
            target_rot = gymapi.Quat.from_euler_zyx(-0.5*math.pi, 0, 0)  # x,y,z
            self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)

            pos_err = self.target_pos - self.hand_pos
            orn_err = orientation_error(self.target_rot, self.hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # unsqueeze(-1) meaning up one dim internal
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            damping = 0.05
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)      ###########17
            self.ur5_dof_targets[:, :6] = self.ur5_dof_pos[:, :6] + u.squeeze(-1) 
            
        if self.progress_buf[i] > 147 and self.progress_buf[i] <= 152:
            self.target_pos = self.hand_pos.clone()
            self.target_pos[:,2] += 0.5
            target_rot = gymapi.Quat.from_euler_zyx(-0.5*math.pi, 0, 0)  # x,y,z
            self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)

            pos_err = self.target_pos - self.hand_pos
            orn_err = orientation_error(self.target_rot, self.hand_rot)

            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # unsqueeze(-1) meaning up one dim internal
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            damping = 0.05
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)      ###########17
            self.ur5_dof_targets[:, :6] = self.ur5_dof_pos[:, :6] + u.squeeze(-1)  

        if self.progress_buf[i] > 152 and self.progress_buf[i] <= 161:
            self.ur5_dof_targets[:, 0]=1
            self.ur5_dof_targets[:, 1]=-1.57
            self.ur5_dof_targets[:, 2]=1.57
            self.ur5_dof_targets[:, 3]=0.
            self.ur5_dof_targets[:, 4]=-1.57
            self.ur5_dof_targets[:, 5]=-1.4177

        if self.progress_buf[i] > 161 and self.progress_buf[i] <= 170:
            self.ur5_dof_targets[:, 0]=-0.9344
            self.ur5_dof_targets[:, 1]=-1.57
            self.ur5_dof_targets[:, 2]=1.57
            self.ur5_dof_targets[:, 3]=0.
            self.ur5_dof_targets[:, 4]=0
            self.ur5_dof_targets[:, 5]=-1.4177
                
        if self.progress_buf[i] > 170 and self.progress_buf[i] <= 182:
            self.ur5_dof_targets[:, 0]=-1.1
            self.ur5_dof_targets[:, 1]=-0.29
            self.ur5_dof_targets[:, 2]=1.5144
            self.ur5_dof_targets[:, 3]=0.3867
            self.ur5_dof_targets[:, 4]=-0.8055
            self.ur5_dof_targets[:, 5]=-1.4177

        if self.progress_buf[i] > 182 and self.progress_buf[i] <= 192:
            self.ur5_dof_targets[:, 0]=-0.7411
            self.ur5_dof_targets[:, 1]=-0.29
            self.ur5_dof_targets[:, 2]=1.5144
            self.ur5_dof_targets[:, 3]=0.3867
            self.ur5_dof_targets[:, 4]=-0.8055
            self.ur5_dof_targets[:, 5]=-1.4177

        if self.progress_buf[i] >192 and self.progress_buf[i] <=276:
            self.target_pos = self.hand_pos.clone()
            self.target_pos[:,0] += self.actions[:,0] *self.dt * self.action_scale
            self.target_pos[:,1] = 0.45
            self.target_pos[:,2] = 0.81125

            limits = torch.ones_like(self.actions[:,1])
            self.target_pos[:,0] = torch.where(self.target_pos[:,0] > 0.9, limits*0.9, self.target_pos[:,0])
            self.target_pos[:,0] = torch.where(self.target_pos[:,0] < 0.4, limits*0.4, self.target_pos[:,0])
            target_rot = gymapi.Quat.from_euler_zyx(0, 0.5*math.pi, 0)  # x,y,z
            self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)
            pos_err = self.target_pos - self.hand_pos
            orn_err = orientation_error(self.target_rot, self.hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # unsqueeze(-1) meaning up one dim internal
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            damping = 0.05 # 0.05
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)      ###########17
            self.ur5_dof_targets[:, :6] = self.ur5_dof_pos[:, :6] + u.squeeze(-1)       # 17
        
        if self.progress_buf[i] > 276 and self.progress_buf[i] <= 279:
            self.ur5_dof_targets[:, 6]=0.4673
            self.ur5_dof_targets[:, 7]=0
            self.ur5_dof_targets[:, 8]=-0.6310
            self.ur5_dof_targets[:, 9]=0

            self.ur5_dof_targets[:, 10]=0.4673
            self.ur5_dof_targets[:, 11]=0
            self.ur5_dof_targets[:, 12]=-0.6310
            self.ur5_dof_targets[:, 13]=0

            self.ur5_dof_targets[:, 14]=0.4673
            self.ur5_dof_targets[:, 15]=0
            self.ur5_dof_targets[:, 16]=-0.6310

        if self.progress_buf[i] > 279 and self.progress_buf[i] <= 282:
            self.ur5_dof_targets[:, 6]=0.7439
            self.ur5_dof_targets[:, 7]=0
            self.ur5_dof_targets[:, 8]=-0.8589
            self.ur5_dof_targets[:, 9]=0

            self.ur5_dof_targets[:, 10]=0.7439
            self.ur5_dof_targets[:, 11]=0
            self.ur5_dof_targets[:, 12]=-0.8589
            self.ur5_dof_targets[:, 13]=0

            self.ur5_dof_targets[:, 14]=0.7439
            self.ur5_dof_targets[:, 15]=0
            self.ur5_dof_targets[:, 16]=-0.8589
           

        if self.progress_buf[i] > 285:
            self.target_pos = self.hand_pos.clone()
            self.target_pos[:,2] = torch.where(self.target_pos[:,2]<1, torch.ones_like(self.target_pos[:,2])*1.05, self.target_pos[:,2])
            self.target_pos[:,0] = torch.where(self.target_pos[:,0]>0.9, torch.ones_like(self.target_pos[:,0])*0.9, self.target_pos[:,0])
            self.target_pos[:,0] = torch.where(self.target_pos[:,0]<0.5, torch.ones_like(self.target_pos[:,0])*0.5, self.target_pos[:,0])
            self.target_pos[:,1] = 0.6 # torch.where(self.target_pos[:,1]<0.4, torch.ones_like(self.target_pos[:,0])*0.45, self.target_pos[:,1])

            target_rot = gymapi.Quat.from_euler_zyx(0, 0.5*math.pi, 0)  # x,y,z
            self.target_rot = torch.tensor([[target_rot.x, target_rot.y, target_rot.z, target_rot.w]] * self.num_envs, dtype=torch.float32).to(self.device)
            pos_err = self.target_pos - self.hand_pos
            orn_err = orientation_error(self.target_rot, self.hand_rot)
            dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)  # unsqueeze(-1) meaning up one dim internal
            j_eef_T = torch.transpose(self.j_eef, 1, 2)
            damping = 0.05
            lmbda = torch.eye(6, device=self.device) * (damping ** 2)
            u = (j_eef_T @ torch.inverse(self.j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)      ###########17
            self.ur5_dof_targets[:, :6] = self.ur5_dof_pos[:, :6] + u.squeeze(-1)       # 17
            
        self.gym.set_dof_position_target_tensor(self.sim,
                                                gymtorch.unwrap_tensor(self.ur5_dof_targets))

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.compute_observations()
        self.compute_reward(self.actions)

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            #self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                # px = (self.rigid_body_states[:, self.hand_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.rigid_body_states[:, self.hand_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.rigid_body_states[:, self.hand_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.hand_handle][:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.rigid_body_states[:, self.hand_handle][:, 0:3][i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                # px = (self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.wrist_2_handle][:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                # py = (self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.wrist_2_handle][:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                # pz = (self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i] + quat_apply(self.rigid_body_states[:, self.wrist_2_handle][:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                # p0 = self.rigid_body_states[:, self.wrist_2_handle][:, 0:3][i].cpu().numpy()
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                # self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.base_entry[:, 0:3][i] + quat_apply(self.base_entry[:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.base_entry[:, 0:3][i] + quat_apply(self.base_entry[:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.base_entry[:, 0:3][i] + quat_apply(self.base_entry[:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.base_entry[:, 0:3][i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

                px = (self.shaft_tail[:, 0:3][i] + quat_apply(self.shaft_tail[:, 3:7][i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                py = (self.shaft_tail[:, 0:3][i] + quat_apply(self.shaft_tail[:, 3:7][i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                pz = (self.shaft_tail[:, 0:3][i] + quat_apply(self.shaft_tail[:, 3:7][i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.shaft_tail[:, 0:3][i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ur5_reward(
    reset_buf, progress_buf, actions, box_pos, hand_pos,
    num_envs, dist_reward_scale, rot_reward_scale, around_handle_reward_scale, open_reward_scale,
    finger_dist_reward_scale, action_penalty_scale, scale, max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor,  int, float, float, float, float, float, float, float, float) -> Tuple[Tensor, Tensor,Tensor]
    x_diff = box_pos[:,0] - hand_pos[:,0]
    x_diff_reward = -2.5 * abs(x_diff)  #  1 / (2 + x_diff ** 2 * 50)

    rewards = x_diff_reward  # + to_edge_reward  # + hand_dist_reward + box_dist_reward
    rewards = torch.where(box_pos[:,2] > 0.83, rewards+5, rewards)
    rewards = torch.where(box_pos[:,2] > 0.7, torch.where(box_pos[:,1] > 0.85, torch.where(box_pos[:,1] < 0.9, rewards+1, rewards), rewards), rewards)   # # and box_pos[:,2] == 0.231 0.3375 < box_pos[:,0] < 0.4625
    rewards = torch.where(box_pos[:,2] > 0.7, torch.where(box_pos[:,1] < 0.84, rewards-1, rewards), rewards)   # # and box_pos[:,2] == 0.231 0.3375 < box_pos[:,0] < 0.4625
    rewards = torch.where(box_pos[:,2] < 0.7, rewards-1, rewards)
    
    success = torch.zeros_like(rewards)
    success_buf = torch.where(box_pos[:,2] > 0.83, torch.where(progress_buf[0] == 318, torch.ones_like(success), success), success)

    reset_buf = torch.where(progress_buf == max_episode_length, torch.ones_like(reset_buf), reset_buf)
    
    return rewards, reset_buf, success_buf

@torch.jit.script
def compute_grasp_transforms(hand_rot, hand_pos, ur5_local_grasp_rot, ur5_local_grasp_pos,
                             drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos
                             ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]

    global_ur5_rot, global_ur5_pos = tf_combine(
        hand_rot, hand_pos, ur5_local_grasp_rot, ur5_local_grasp_pos)
    global_drawer_rot, global_drawer_pos = tf_combine(
        drawer_rot, drawer_pos, drawer_local_grasp_rot, drawer_local_grasp_pos)

    return global_ur5_rot, global_ur5_pos, global_drawer_rot, global_drawer_pos

@torch.jit.script
def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)