import numpy as np
import os
import torch
import random
import glob
from scipy.spatial.transform import Rotation as R

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from tasks.hand_base.base_task import BaseTask

def iprint(*strings):
    print(strings)
    exit()

class Rot(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg

        self.sim_params = sim_params
        self.physics_engine = physics_engine

        num_states = 0

        self.cfg["env"]["numObservations"] = 13
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 27

        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        super().__init__(cfg=self.cfg)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1, 0, 1.5)
            cam_target = gymapi.Vec3(0.2, 0.2, 0.75)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.dt = self.sim_params.dt
        
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3].clone()
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7].clone()
        
        self.object_rot_mod = torch.tensor([0,0,0.999,0.008],device=self.device).repeat(self.num_envs, 1)
        self.object_axes_mod = torch.tensor([0,0.707,0,0.707],device=self.device).repeat(self.num_envs, 1)
        
        self.grasp_list = []
        grasps = np.load('/home/smondal/Desktop/DexterousHands/affordpose_data/Dataset/Single/validated/filtered_grasps.npy')
        self.grasp_list = torch.from_numpy(grasps).to(self.device)
        self.grasp_list = self.grasp_list[:self.num_envs,:]

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, "z")
        self.sim_params.physx.max_gpu_contact_pairs = int(self.sim_params.physx.max_gpu_contact_pairs)
        self.sim_params.gravity.z = -9.8 

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

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        
        table_dims = gymapi.Vec3(1, 1, 0.3)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table_asset_2 = self.gym.create_box(self.sim, table_dims.x/2, table_dims.y/2, table_dims.z, table_asset_options)
        
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0, 0.15)
        table_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        table_pose_2 = gymapi.Transform()
        table_pose_2.p = gymapi.Vec3(0.25, 0, 0.45)
        table_pose_2.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        object_asset_root = "/home/smondal/Desktop/DexterousHands/affordpose_data/Dataset/Single/urdf"
        object_file_name = "AffordPose_mug_8619_1000.urdf"

        self.num_object_bodies = 0
        self.num_object_shapes = 0
        
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.vhacd_enabled = True
        object_asset_options.fix_base_link = True
        object_asset_options.armature = 0.025
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.vhacd_params = gymapi.VhacdParams()

        object_asset = self.gym.load_asset(self.sim, object_asset_root, object_file_name, object_asset_options)

        self.num_object_bodies += self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes += self.gym.get_asset_rigid_shape_count(object_asset)
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.25,0,0.675) 
        object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 3.14)
               
        max_agg_bodies =  100
        max_agg_shapes =  100
        
        self.envs = []

        self.table_indices = []
        self.object_indices = []

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            table_2_handle = self.gym.create_actor(env_ptr, table_asset_2, table_pose_2, "table2", i, -1, 0)

            actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            
            for actor_shape_prop in actor_shape_props:
                actor_shape_prop.friction = 10
            
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, actor_shape_props)

            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)
           
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, -1, 1)

            actor_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            
            for actor_shape_prop in actor_shape_props:
                actor_shape_prop.friction = 10
            
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, actor_shape_props)

            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)

            self.object_indices.append(object_idx)

            self.envs.append(env_ptr)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)

    def compute_observations(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
   
        grasp_pose_r = robust_compute_rotation_matrix_from_ortho6d(self.grasp_list[:, 3:9].cpu())
        grasp_pose_T = torch.zeros(self.num_envs, 4, 4).to(self.device)
        grasp_pose_T[:,:3,:3] = grasp_pose_r
        grasp_pose_T[:,3, 3] = 1
        grasp_pose_T[:,:3, 3] = self.grasp_list[:, 0:3]

        obj_pose_r = quaternion_to_rotation_matrix(self.object_rot)
        obj_pose_T = torch.zeros(self.num_envs, 4, 4).to(self.device)
        obj_pose_T[:,:3,:3] = obj_pose_r.to(self.device)
        obj_pose_T[:,3, 3] = 1
        obj_pose_T[:,:3, 3] = self.object_pos

        grasp_pose_world_T = obj_pose_T @ grasp_pose_T
               
        axes_T = torch.tensor(np.array([    [0.0000000,  1.0000000,  0.0000000, 0],
                                            [-1.0000000,  0.0000000,  0.0000000, 0],
                                            [0.0000000,  0.0000000,  1.0000000, 0],
                                            [0, 0, 0, 1]
                                            ]),dtype=torch.float).to(self.device)
                                                            
        axes_T = axes_T.unsqueeze(0)
        axes_T = axes_T.repeat(self.num_envs, 1, 1)
          
        grasp_pose_world_T = grasp_pose_world_T @ axes_T

        grasp_pose_world_quat = torch.zeros(self.num_envs, 4)
        for index, r in enumerate(grasp_pose_world_T[:,:3,:3].cpu().numpy()):
            obj_r = R.from_matrix(r)
            obj_quat = obj_r.as_quat()  # XYZW
            grasp_pose_world_quat[index] = torch.from_numpy(obj_quat)
                
        self.object_pos = grasp_pose_world_T[:,:3,3]
        self.object_rot = grasp_pose_world_quat
 
    def pre_physics_step(self, actions):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.actions = actions.clone().to(self.device)

        original_rot = self.root_state_tensor[self.object_indices[0], 3:7]

        if self.progress_buf[0] % 30 == 0:
            
            self.root_state_tensor[self.object_indices[0], 3:7] += quat_mul(self.object_rot_mod.squeeze(0), original_rot) 

            object_indices = torch.unique(self.object_indices[0]).to(torch.int32)

            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_state_tensor),
                                                        gymtorch.unwrap_tensor(object_indices), len(object_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()

        if True:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
       
    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

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

def cross_product(u, v):
    batch = u.shape[0]
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
        
    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)
        
    return out

def normalize_vector(v):
    batch = v.shape[0]
    v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    v_mag = torch.max(v_mag, v.new([1e-8]))
    v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    v = v/v_mag
    return v

def robust_compute_rotation_matrix_from_ortho6d(poses):
    """
    Instead of making 2nd vector orthogonal to first
    create a base that takes into account the two predicted
    directions equally
    """
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    y = normalize_vector(y_raw)  # batch*3
    middle = normalize_vector(x + y)
    orthmid = normalize_vector(x - y)
    x = normalize_vector(middle + orthmid)
    y = normalize_vector(middle - orthmid)
    # Their scalar product should be small !
    # assert torch.einsum("ij,ij->i", [x, y]).abs().max() < 0.00001
    z = normalize_vector(cross_product(x, y))

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    # Check for reflection in matrix ! If found, flip last vector TODO
    # assert (torch.stack([torch.det(mat) for mat in matrix ])< 0).sum() == 0
    return matrix

def quaternion_to_rotation_matrix(quaternions):
    """
    Convert a tensor of quaternions to a tensor of 3x3 rotation matrices.
    
    Args:
        quaternions (torch.Tensor): Tensor of shape (N, 4), where N is the number of quaternions, 
                                    and each quaternion is represented as (x, y, z, w).
                                    
    Returns:
        torch.Tensor: Tensor of shape (N, 3, 3) containing the corresponding 3x3 rotation matrices.
    """
    # Ensure the quaternions are normalized
    quaternions = quaternions / quaternions.norm(dim=1, keepdim=True)
    
    # Extract individual quaternion components
    x, y, z, w = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]

    # Compute the terms used in the rotation matrix
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    # Create the rotation matrix
    rotation_matrix = torch.zeros((quaternions.shape[0], 3, 3), device=quaternions.device)

    rotation_matrix[:, 0, 0] = ww + xx - yy - zz
    rotation_matrix[:, 0, 1] = 2 * (xy - wz)
    rotation_matrix[:, 0, 2] = 2 * (xz + wy)
    
    rotation_matrix[:, 1, 0] = 2 * (xy + wz)
    rotation_matrix[:, 1, 1] = ww - xx + yy - zz
    rotation_matrix[:, 1, 2] = 2 * (yz - wx)
    
    rotation_matrix[:, 2, 0] = 2 * (xz - wy)
    rotation_matrix[:, 2, 1] = 2 * (yz + wx)
    rotation_matrix[:, 2, 2] = ww - xx - yy + zz

    return rotation_matrix