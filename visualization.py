import matplotlib.pyplot as plt
import numpy as np
import os

def get_c2w_poses(datapath, mode='train'):
    '''
    Given a folder full of txt files containing camera to world poses, puts all into a single variable
        
        INPUT: datapath to folder containing all training and test data in correct format
        OUTPUT: array of poses, one for each image
    '''
    pose_file_names = [f for f in os.listdir(datapath + f'/{mode}/pose') if f.endswith('.txt')]
    pose_file_names = sorted(pose_file_names, key=lambda x: str(x.split('.')[0]))
    intrinsics_file_names = [f for f in os.listdir(datapath + f'/{mode}/intrinsics') if f.endswith('.txt')]
    intrinsics_file_names = sorted(intrinsics_file_names, key=lambda x: str(x.split('.')[0]))
    
    assert len(pose_file_names) == len(intrinsics_file_names) # sanity check
    
    # Read
    N = len(pose_file_names)
    poses = np.zeros((N,4,4))
    
    for i in range(N):
        name = pose_file_names[i]
        pose = open(datapath + f'/{mode}/pose/' + name).read().split()
        poses[i] = np.array(pose, dtype=float).reshape(4,4)

    return poses


def visualize_camera_poses(camera_poses):
    '''
    Visualizes the camera poses present after calling get_c2w_poses
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, pose in enumerate(camera_poses):
        position = pose[:3, 3]
        forward_direction = pose[:3, 2]  # Assuming the third column represents the forward direction

        # Plotting camera position
        ax.scatter(position[0], position[1], position[2], c='blue', marker='o')

        # Plotting camera direction as an arrow
        ax.quiver(position[0], position[1], position[2], -forward_direction[0], -forward_direction[1], -forward_direction[2], color='red', length=10, normalize=True)

    ax.set_xlim(-11,11)
    ax.set_ylim(-11,11)
    ax.set_zlim(-11,11)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def visualize_rays(origins, directions, num_rays_to_sample_per_set=None):
    '''
    Visualizes the camera rays
    '''
    num_sets, num_rays, _ = origins.shape

    if num_rays_to_sample_per_set is not None:
        # Sample a subset of rays for each set
        sampled_indices = np.random.choice(num_rays, num_rays_to_sample_per_set, replace=False)

        # Create a mask for indexing
        mask = np.zeros((num_rays,), dtype=bool)
        mask[sampled_indices] = True

        # Apply the mask to each set
        sampled_origins = origins[:, mask, :]
        sampled_directions = directions[:, mask, :]
    else:
        sampled_origins = origins
        sampled_directions = directions

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the ray origins for each set
    for i in range(num_sets):
        ax.scatter(sampled_origins[i, :, 0], sampled_origins[i, :, 1], sampled_origins[i, :, 2], marker='o')

    # Plotting the ray directions as arrows for each set
    for i in range(num_sets):
        for j in range(sampled_directions.shape[1]):
            origin = sampled_origins[i, j, :]
            direction = sampled_directions[i, j, :]
            ax.quiver(origin[0], origin[1], origin[2], -direction[0], -direction[1], -direction[2], color='red', length=10, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()