#!/usr/bin/env python
"""
AUTHOR: Matthieu Marchal (SII Internship)
LAST UPDATED: 2024-04-04
DESCRIPTION: 
    This script filters images based on distance and spherical coverage, 
    and then finds the shortest path through the selected images using Cartesian coordinates.
    It aims to improve accuracy in selecting images corresponding to spherical coverage
    and optimize the shortest path calculation.

USAGE:
    python3 filter_path.py

COMMENTS: 
    ! Azimuth and elevation are not used for now, but they could be used improve coverage selection. ! 

CONFIGURATION : 
    All configurations are adjustable in the "CONFIGURATION" section below.
"""
import numpy as np
import json
import os
from matplotlib import pyplot as plt
from PIL import Image
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### CONFIGURATION ###
# Base directory for data storage
RELATIVE_ROOT = '../..'
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), RELATIVE_ROOT))

# Paths and parameters configuration
GROUND_TRUTH_FILE = 'data/src/speedplusv2/synthetic/train.json'
IMAGE_FOLDER = 'data/src/speedplusv2/synthetic/images'
SAVE_FOLDER = 'data/src/sfm/synthetic/satellite_2x2'

# Convert paths to absolute paths
GROUND_TRUTH_PATH = os.path.join(ROOT_PATH, GROUND_TRUTH_FILE)
IMAGE_PATH = os.path.join(ROOT_PATH, IMAGE_FOLDER)
SAVE_PATH = os.path.join(ROOT_PATH, SAVE_FOLDER)

# Parameters for filtering and processing images
AZIMUTH_SEGMENTS = 2
ELEVATION_SEGMENTS = 2
DISTANCE_RANGE = [4, 5]  # The range of distances to filter images
### END OF CONFIGURATION ###

def loadJson(file_path):
    """Load a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def toSphericalCoordinates(q):
    """Convert a quaternion to spherical coordinates (azimuth, elevation)."""
    euler = R.from_quat(q).as_euler('zyx', degrees=True)
    azimuth = np.radians(euler[0])
    elevation = np.radians(euler[1])
    return azimuth, elevation

def getCartesianCoordinates(azimuth, elevation, r=1):
    """Convert spherical coordinates to Cartesian coordinates."""
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])

def filterByDistanceAndSphericalCoverage(data, distance_range):
    """Filter images based on distance and spherical coverage."""
    segments = {(az, el): [] for az in range(AZIMUTH_SEGMENTS) for el in range(ELEVATION_SEGMENTS)}
    
    for item in data:
        image_full_path = os.path.join(IMAGE_PATH, item["filename"])
        if os.path.exists(image_full_path):
            r = np.array(item["r_Vo2To_vbs_true"])
            distance = np.linalg.norm(r)
            if distance_range[0] <= distance <= distance_range[1]:
                q = item["q_vbs2tango_true"]
                azimuth, elevation = toSphericalCoordinates(q)
                az_index = int(((azimuth) % (2 * np.pi)) / ((2 * np.pi) / AZIMUTH_SEGMENTS))
                el_index = int((elevation + np.pi) / (2 * np.pi / ELEVATION_SEGMENTS))
                segment_key = (az_index, el_index)
                segments[segment_key].append((item["filename"], q, r))

    selected_images = [img for segment_images in segments.values() for img in segment_images if len(segment_images) > 0]
    return selected_images

def findShortestPath(selected_images):
    """Find the shortest path through selected images using a greedy approach."""
    points = np.array([getCartesianCoordinates(*toSphericalCoordinates(q)) for _, q, _ in selected_images])

    distance_matrix = cdist(points, points)
    
    num_images = len(selected_images)
    visited = [False] * num_images
    path = [0]
    visited[0] = True

    for _ in range(1, num_images):
        last = path[-1]
        next_point = np.argmin([distance_matrix[last][j] if not visited[j] else np.inf for j in range(num_images)])
        path.append(next_point)
        visited[next_point] = True
    return path

def saveImages(selected_images, save_path, tsp_path):
    """Save the selected images in TSP order."""
    os.makedirs(save_path, exist_ok=True)
    for i, index in enumerate(tsp_path):
        filename, _, _ = selected_images[index]
        img_path = os.path.join(IMAGE_PATH, filename)
        img = Image.open(img_path)
        img.save(os.path.join(save_path, f"{i:03d}_{filename}"))

    print(f"Images saved to {save_path} in TSP order")

def main():
    """Main function to filter and process images."""
    data = loadJson(GROUND_TRUTH_PATH)
    selected_images = filterByDistanceAndSphericalCoverage(data, DISTANCE_RANGE)
    tsp_path = findShortestPath(selected_images)
    saveImages(selected_images, SAVE_PATH, tsp_path)
    print(f"Processed {len(selected_images)} images.")

if __name__ == "__main__":
    main()
    