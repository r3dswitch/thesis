from isaacgym.torch_utils import *
from scipy.spatial.transform import Rotation as R

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):
    damping = 0.05
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

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

@torch.jit.script
def randomize_rotation(rand, unit_tensor, degree: int):
    mult = degree / 180
    return quat_from_angle_axis(rand * np.pi * mult, unit_tensor)

def compute_world_pose(obj, matrix_list, object_rot, object_pos):
    """
    Computes the rotated pose in the world frame based on rotation and object pose.

    Parameters:
    - matrix_list (torch.Tensor): List of transfromation matrices wrt object frame
    - object_rot (torch.Tensor): Quaternion representing the object's rotation wrt world frame
    - object_pos (torch.Tensor): Position of the object wrt world frame
    
    """
    # Compute relative pose rotation matrix
    relative_pose_r = robust_compute_rotation_matrix_from_ortho6d(matrix_list[:, 3:9].cpu())
    relative_pose_T = torch.zeros(obj.num_envs, 4, 4).to(obj.device)
    relative_pose_T[:, :3, :3] = relative_pose_r
    relative_pose_T[:, 3, 3] = 1
    relative_pose_T[:, :3, 3] = matrix_list[:, 0:3]

    # Compute object pose rotation matrix
    obj_pose_r = quaternion_to_rotation_matrix(object_rot)
    obj_pose_T = torch.zeros(obj.num_envs, 4, 4).to(obj.device)
    obj_pose_T[:, :3, :3] = obj_pose_r.to(obj.device)
    obj_pose_T[:, 3, 3] = 1
    obj_pose_T[:, :3, 3] = object_pos

    # Calculate relative pose in world frame
    relative_pose_world_T = obj_pose_T @ relative_pose_T

    # Convert rotation matrices to quaternions
    relative_pose_world_quat = torch.zeros(obj.num_envs, 4)
    for index, r in enumerate(relative_pose_world_T[:, :3, :3].cpu().numpy()):
        obj_r = R.from_matrix(r)
        obj_quat = obj_r.as_quat()  # XYZW
        relative_pose_world_quat[index] = torch.from_numpy(obj_quat)

    # Set the computed relative position and rotation attributes
    relative_pos = relative_pose_world_T[:, :3, 3].to(obj.device)
    relative_rot = relative_pose_world_quat.to(obj.device)

    return relative_pos, relative_rot