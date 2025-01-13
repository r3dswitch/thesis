while not env.gym.query_viewer_has_closed(env.viewer):

                if object_stablity:

                    batch_object_stable_pose = batch_object_pose.clone()
                   
                    hand_pose_rot_mtx = robust_compute_rotation_matrix_from_ortho6d(hand_pose[:,3:9])
                    hand_pose_transformation_mtx = torch.zeros(len(env_ids), 4, 4).to(env.device)
                    hand_pose_transformation_mtx[:,:3,:3] = hand_pose_rot_mtx
                    hand_pose_transformation_mtx[:,3, 3] = 1
                    hand_pose_transformation_mtx[:,:3, 3] = hand_pose[:,0:3]

                    obj_pose_rot_mtx_tensor = torch.zeros(len(env_ids), 3, 3)
                    for index, object_pose_i_quat in enumerate(batch_object_pose[:,3:7].cpu().numpy()):
                        obj_rotation_mtx_i = R.from_quat(object_pose_i_quat)
                        obj_pose_rot_mtx_tensor[index] = torch.from_numpy(obj_rotation_mtx_i.as_matrix())

                    obj_pose_transformation_mtx = torch.zeros(len(env_ids), 4, 4).to(env.device)
                    obj_pose_transformation_mtx[:,:3,:3] = obj_pose_rot_mtx_tensor.to(env.device)
                    obj_pose_transformation_mtx[:,3, 3] = 1
                    obj_pose_transformation_mtx[:,:3, 3] = batch_object_pose[:,0:3]

                    hand_pose_transformation_mtx_world = obj_pose_transformation_mtx @ hand_pose_transformation_mtx
               
                    transform_mtx_z_minus_90 = torch.tensor(np.array([  [0.0000000,  1.0000000,  0.0000000, 0],
                                                                        [-1.0000000,  0.0000000,  0.0000000, 0],
                                                                        [0.0000000,  0.0000000,  1.0000000, 0],
                                                                        [0, 0, 0, 1]
                                                                        ]),dtype=torch.float).to(env.device)
                    transform_mtx_z_minus_90 = transform_mtx_z_minus_90.unsqueeze(0)
                    batch_transform_mtx_z_minus_90 = transform_mtx_z_minus_90.repeat(len(env_ids), 1, 1)
          
                    hand_pose_transformation_mtx_world_isaac = hand_pose_transformation_mtx_world @ batch_transform_mtx_z_minus_90

                    hand_pose_quat_world = torch.zeros(len(env_ids), 4)
                    for index, hand_pose_rot_mtx_i in enumerate(hand_pose_transformation_mtx_world_isaac[:,:3,:3].cpu().numpy()):
                    
                        obj_rotation_mtx_i = R.from_matrix(hand_pose_rot_mtx_i)
                        obj_rotation_quat_i = obj_rotation_mtx_i.as_quat()  # XYZW
                        
                        hand_pose_quat_world[index] = torch.from_numpy(obj_rotation_quat_i)
                    
                    if criterion_mse(env.get_observation(), batch_object_stable_pose) < 0.1 and not place_hand_into_object_frame:
                        env.reset_agent_pose(env_ids, env.indice_hand, hand_pose_transformation_mtx_world_isaac[:,0:3, 3], hand_pose_quat_world.to(env.device))
                        place_hand_into_object_frame = True
                    
                    above_orientation_list, above_position_list = check_hithand_base_pose_batch(hand_pose_transformation_mtx_world_isaac)
                    
                    pose_valid_list = []
                    for orien_valid, position_valid in zip(above_orientation_list, above_position_list):
                        pose_valid_list.append(orien_valid * position_valid)
                    assert len(pose_valid_list) == num_envs