from isaacgym import gymapi, gymtorch
import torch, glob, os
from isaacgym.torch_utils import *

class IsaacFactory:
    @staticmethod
    def add_debug_lines(obj, env, pos, rot):
            posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=obj.device) * 0.2)).cpu().numpy()
            posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=obj.device) * 0.2)).cpu().numpy()
            posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=obj.device) * 0.2)).cpu().numpy()

            p0 = pos.cpu().numpy()
            obj.gym.add_lines(obj.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
            obj.gym.add_lines(obj.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
            obj.gym.add_lines(obj.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])
    
    @staticmethod        
    def create_sim(obj):
        obj.dt = obj.sim_params.dt 
        obj.up_axis_idx = obj.set_sim_params_up_axis(obj.sim_params, obj.up_axis)
        obj.sim_params.physx.max_gpu_contact_pairs = int(obj.sim_params.physx.max_gpu_contact_pairs)
        obj.sim_params.gravity.z = obj.cfg["env"]['gravity'] 

        obj.sim = super().create_sim(obj.device_id, obj.graphics_device_id, obj.physics_engine, obj.sim_params)
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        obj.gym.add_ground(obj.sim, plane_params)
        obj._create_envs(obj.num_envs, obj.cfg["env"]['envSpacing'], int(np.sqrt(obj.num_envs)))
        
        if obj.randomize:
            obj.apply_randomizations(obj.randomization_params)
    
    @staticmethod        
    def refresh_tensors(obj):
        obj.gym.refresh_dof_state_tensor(obj.sim)
        obj.gym.refresh_actor_root_state_tensor(obj.sim)
        obj.gym.refresh_rigid_body_state_tensor(obj.sim)
        obj.gym.refresh_net_contact_force_tensor(obj.sim)
        obj.gym.refresh_jacobian_tensors(obj.sim)
    
    @staticmethod    
    def load_hand_asset(obj):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), obj.cfg["env"]["asset"]["handAssetRoot"])
        hand_asset_file = obj.cfg["env"]["asset"]["handAssetFileName"]

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
        table_dims = gymapi.Vec3(1, 1, 0.5)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table1_asset = obj.gym.create_box(obj.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table2_asset = obj.gym.create_box(obj.sim, table_dims.x / 4, table_dims.y / 4, table_dims.z, table_asset_options)
        return table1_asset, table2_asset, table_dims

    @staticmethod
    def load_object_asset(obj):
        object_asset_root = obj.cfg["env"]["asset"]["objectAssetRoot"]
        object_file_name = obj.cfg["env"]["asset"]["objectAssetFileName"]
        
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.fix_base_link = obj.cfg["env"]["asset"]["objectAssetOptions"]["fixBaseLink"]
        object_asset_options.vhacd_enabled = True
        object_asset_options.armature = 0.025
        object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
        object_asset_options.vhacd_params = gymapi.VhacdParams()
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        object_asset = obj.gym.load_asset(obj.sim, object_asset_root, object_file_name, object_asset_options)
        return object_asset
    
    @staticmethod
    def initialize_robot_dof_props(gym, hand_asset):
            robot_dof_props = gym.get_asset_dof_properties(hand_asset)
            robot_lower_qpos, robot_upper_qpos = [], []

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

            return robot_dof_props, robot_lower_qpos, robot_upper_qpos
    
    @staticmethod
    def initialize_poses(table_dims):
        table1_pose = gymapi.Transform()
        table1_pose.p = gymapi.Vec3(0.0, 0, table_dims.z * 0.5)
        table1_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
            
        table2_pose = gymapi.Transform()
        table2_pose.p = gymapi.Vec3(table_dims.x * 0.2, 0, table_dims.z)
        table2_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)
        
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(table_dims.x * 0.2, 0, 0.85)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 3.14)

        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(table_dims.x * -0.4, 0, 0.5)  # Based on table height
        hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0.0)

        return table1_pose, table2_pose, object_start_pose, hand_start_pose

    @staticmethod
    def env_initialisation(obj):
        obj.envs = []
        obj.object_init_states = []
        obj.hand_init_states = []
        obj.hand_indices = []
        obj.fingertip_indices = []
        obj.table_indices = []
        obj.object_indices = []
    
    @staticmethod    
    def env_append(obj, env_ptr, hand_start_pose, object_start_pose, hand_actor, table1_actor, table2_actor, object_actor):
        obj.envs.append(env_ptr)
        
        obj.hand_init_states.append([   hand_start_pose.p.x,
                                        hand_start_pose.p.y,
                                        hand_start_pose.p.z,
                                        hand_start_pose.r.x,
                                        hand_start_pose.r.y,
                                        hand_start_pose.r.z,
                                        hand_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        obj.object_init_states.append([ object_start_pose.p.x, 
                                        object_start_pose.p.y, 
                                        object_start_pose.p.z,
                                        object_start_pose.r.x, 
                                        object_start_pose.r.y, 
                                        object_start_pose.r.z, 
                                        object_start_pose.r.w,
                                        0, 0, 0, 0, 0, 0] )
        
        hand_idx = obj.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
        table1_idx = obj.gym.get_actor_index(env_ptr, table1_actor, gymapi.DOMAIN_SIM)
        table2_idx = obj.gym.get_actor_index(env_ptr, table2_actor, gymapi.DOMAIN_SIM)
        object_idx = obj.gym.get_actor_index(env_ptr, object_actor, gymapi.DOMAIN_SIM)
        
        obj.hand_indices.append(hand_idx)
        obj.table_indices.append(table1_idx)
        obj.table_indices.append(table2_idx)
        obj.object_indices.append(object_idx)
    
    @staticmethod    
    def make_tensors(obj):
        obj.hand_init_states = to_torch(obj.hand_init_states, device=obj.device).view(obj.num_envs, 13)
        obj.object_init_states = to_torch(obj.object_init_states, device=obj.device).view(obj.num_envs, 13)
        obj.hand_indices = to_torch(obj.hand_indices, dtype=torch.long, device=obj.device)
        obj.object_indices = to_torch(obj.object_indices, dtype=torch.long, device=obj.device)
        obj.lower_limits = torch.tensor(obj.lower_limits, device=obj.device)
        obj.upper_limits = torch.tensor(obj.upper_limits, device=obj.device)
    
    @staticmethod    
    def iprint(*strings):
        print(strings)
        exit()
    
    @staticmethod    
    def make_observations(obj, obs_name, obs_rigid_body_index):
        setattr(obj, f"{obs_name}_pos", obj.rigid_body_states[:, obs_rigid_body_index, 0:3])
        setattr(obj, f"{obs_name}_rot", obj.rigid_body_states[:, obs_rigid_body_index, 3:7])
        setattr(obj, f"{obs_name}_linvel", obj.rigid_body_states[:, obs_rigid_body_index, 7:10])
        setattr(obj, f"{obs_name}_angvel", obj.rigid_body_states[:, obs_rigid_body_index, 10:13])
    
    @staticmethod    
    def add_displacement(obj, obs_name, axis, dist):
        if axis == 'x' or axis == 'X':
            vec = [1,0,0]
        elif axis == 'y' or axis == 'Y':
            vec = [0,1,0]
        elif axis == 'z' or axis == 'Z':
            vec = [0,0,1]
        else:
            iprint("Error: wrong axis input")
        
        pos = getattr(obj, f"{obs_name}_pos")
        rot = getattr(obj, f"{obs_name}_rot")
        
        setattr(obj, f"{obs_name}_pos", pos + quat_apply(rot, to_torch(vec, device=obj.device).repeat(obj.num_envs, 1) * dist))
