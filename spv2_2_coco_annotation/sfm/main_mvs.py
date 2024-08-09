#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""
MODIFIED BY: Matthieu Marchal (SII Internship)
LAST UPDATED: 2024-04-04
DESCRIPTION: 
    This script performs the Multi-View Stereo (MVS) pipeline from openMVG+openMVS libraries on a set of images.
COMMENTS: 
    Please refer to the official openMVG and openMVS github/documentation for more information.
"""

import os
import subprocess
import sys

### CONFIGURATION ###
# Base directory for data storage relative to the script's location
RELATIVE_ROOT = '../..'
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), RELATIVE_ROOT))

# Paths and parameters configuration
SOURCE_FOLDER = ''
DESTINATION_FOLDER = ''
OPENMVS_BIN_FOLDER = 'openMVS/make/bin' # Folder containing openMVS binaries

# Convert paths to absolute paths
SOURCE_PATH = os.path.join(ROOT_PATH, SOURCE_FOLDER)
DESTINATION_PATH = os.path.join(ROOT_PATH, DESTINATION_FOLDER)
OPENMVS_BIN_PATH = os.path.join(ROOT_PATH, OPENMVS_BIN_FOLDER)

### END OF CONFIGURATION ###

def runOpenmvgOpenmvs(input_dir, output_dir):
    # Prepare the commands
    commands = [
        ["openMVG_main_openMVG2openMVS", "-i", os.path.join(input_dir, "reconstruction_sequential/sfm_data.bin"), "-o", "scene.mvs"],
        [os.path.join(OPENMVS_BIN_PATH, "DensifyPointCloud"), "scene.mvs"],
        [os.path.join(OPENMVS_BIN_PATH, "ReconstructMesh"), "scene_dense.mvs"],
        [os.path.join(OPENMVS_BIN_PATH, "RefineMesh"), "scene_dense_mesh.mvs"],
        [os.path.join(OPENMVS_BIN_PATH, "TextureMesh"), "scene_dense_mesh_refine.mvs"],
        [os.path.join(OPENMVS_BIN_PATH, "Viewer"), "scene_dense_mesh_refine_texture.mvs"]
    ]
    
    # Execute each command
    for command in commands:
        print(f"Executing: {' '.join(command)}")
        p = subprocess.Popen(command, cwd=output_dir)
        p.wait()

if __name__ == "__main__":
    # Now using absolute paths defined in the configuration section
    input_dir = SOURCE_PATH
    output_dir = DESTINATION_PATH

    print("Using input directory: ", input_dir)
    print("Using output directory: ", output_dir)

    runOpenmvgOpenmvs(input_dir, output_dir)