#!/usr/bin/env python3
"""
AUTHOR: Matthieu Marchal (SII Internship)
LAST UPDATED: 2024-04-04
DESCRIPTION: 
    This script runs the custom SfM_GlobalPipeline.py on a set of images.
    
COMMENTS: 
    Please refer to the official openMVG github/documentation for more information.
"""

import subprocess
import sys
import os

# Base directory for data storage
RELATIVE_ROOT = '../..'
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), RELATIVE_ROOT))

### MODIFY BELOW ###
source_folder = 'data/src/speedplusv2/lightbox/images'
destination_folder = 'spv2_annotation/blender/exports/openMVG_reconstruction/lightbox'

f_param = '2988.46'
### END MODIFY ###

def runCommand(image_path, reconstruction_path):
    # Directory where to run the command
    directory = os.path.join(ROOT_PATH, "src/sfm")

    command = [
        "python3", "sfm_global_pipeline.py",
        image_path, reconstruction_path, f_param
    ]
   
    try:
        result = subprocess.run(command, cwd=directory, check=True, text=True)
        print("Command executed successfully")
    except subprocess.CalledProcessError as e:
        print("Error in executing command:", e)
        print("Error Output:", e.stderr)

if __name__ == "__main__":

    image_path = os.path.join(ROOT_PATH, source_folder)
    reconstruction_path = os.path.join(ROOT_PATH, destination_folder)

    print("Using input directory: ", image_path)
    print("Using output directory: ", reconstruction_path)

    runCommand(image_path, reconstruction_path)
