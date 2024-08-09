"""
MODIFIED BY: Matthieu Marchal (SII Internship)
LAST UPDATED: 2024-04-04
DESCRIPTION: 
    This script calculates the rotational error between the ground truth and the estimated rotations from the SfM pipeline.
COMMENTS: 
    To get the estimated rotations you need to export the results from the SfM pipeline in a JSON file. (openMVG_main_ExportsSfM_Data)

MODIFICATIONS: 
    To specify the path of the different folders, please refer to the section "### MODIFY BELOW ###".
"""

import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### MODIFY BELOW ###
ground_truth_file = ''
sfm_results_file =''
###


def findRotationInGroundTruth(item):
    q_vbs2tango_true = item['q_vbs2tango_true']
    rotation_matrix = R.from_quat(q_vbs2tango_true).as_matrix()
    return rotation_matrix

def findRotationInResults(filename, views, extrinsics):
    for view in views:
        if view['value']['ptr_wrapper']['data']['filename'] == filename:
            key = view.get('key')
            for extrinsic in extrinsics:
                if extrinsic['key'] == key:
                    rotation_matrix = extrinsic['value']['rotation']
                    rotation_matrix=R.from_matrix(rotation_matrix).as_matrix()
                    return rotation_matrix

def rotationMatrixAverage(rotation_matrices):
    if len(rotation_matrices) == 0:
        return None
    A = np.sum([np.outer(r[:, i], r[:, i]) for r in rotation_matrices for i in range(3)], axis=0)
    U, _, Vt = np.linalg.svd(A)
    mean_rotation = np.dot(U, Vt)
    return mean_rotation

def visualizeRotations(relative_rotations,mean_global_rotation, filenames):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('3D Relative Rotations Visualization')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Draw the original axes
    ax.quiver(0, 0, 0, 1, 0, 0, color='r', arrow_length_ratio=0.1, label='X-axis')
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', arrow_length_ratio=0.1, label='Y-axis')
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', arrow_length_ratio=0.1, label='Z-axis')

    # Generate random colors for each arrow
    colors = [mcolors.to_hex(np.random.rand(3,)) for _ in relative_rotations]

    # Draw the relative rotations
    for rotation_matrix, filename,colors in zip(relative_rotations, filenames,colors):
        # Apply rotation to the unit vector along the x-axis
        vector = np.dot(rotation_matrix, [1, 0, 0])
        ax.quiver(0, 0, 0, *vector,color=colors, arrow_length_ratio=0.1, label=f'{filename} Relative')

    # Draw the mean global rotation
    vector = np.dot(mean_global_rotation, [1, 0, 0])
    ax.quiver(0, 0, 0, *vector, arrow_length_ratio=0.1, label='Mean Global')
    ax.legend()
    plt.show()

def main(ground_truth_file, results_file):
    def loadJson(file):
        with open(file) as f:
            data = json.load(f)
        return data
    ground_truth_data = loadJson(ground_truth_file)
    results_data = loadJson(results_file)

    views = results_data['views']
    extrinsics = results_data['extrinsics']
    
    relative_rotations = []
    filenames = []
    m=0
    
    # Calculate the relative rotations
    for item in ground_truth_data:
        filename = item['filename']
        filenames.append(filename)
        ground_truth_rotation = findRotationInGroundTruth(item)
        result_rotation = findRotationInResults(filename, views, extrinsics)
        
        if ground_truth_rotation is not None and result_rotation is not None:
            print(f'found matched filename: {filename}')
            m+=1
            relative_rotation = ground_truth_rotation.T@result_rotation
            relative_rotations.append(relative_rotation)
    print(f"Number of matching poses: {m}")
    
    # Calculate the mean global rotation from the relative rotations
    mean_global_rotation = rotationMatrixAverage(np.array(relative_rotations))

    # Print Euler angles of the mean global rotation
    euler_mean_global = R.from_matrix(mean_global_rotation).as_euler('xyz', degrees=True)
    print(f"Mean Global Rotation (Euler XYZ): {euler_mean_global}")
    
    # Visualize the rotations
    visualizeRotations(relative_rotations,mean_global_rotation, filenames)

if __name__ == "__main__":


    ground_truth_relative_path = '../data'
    results_relative_path = '../sfm'
    ground_truth_path= os.path.join(ground_truth_relative_path, ground_truth_file)
    save_path = os.path.join(results_relative_path, sfm_results_file)

    main(ground_truth_file, sfm_results_file)
