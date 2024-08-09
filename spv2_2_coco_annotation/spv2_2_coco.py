#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUTHOR: Matthieu Marchal (SII Internship)
LAST MODIFIED: 2024-07-15

DESCRIPTION: This script converts the SPEED+V2 dataset to the COCO format.
It utilizes Blender to render images and annotations.

NOTE: This script is specific to the SPEED+V2 dataset format.
"""

import sys
import os
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spv2_2_coco_annotation.utils.data_parser import DatasetParser

### PARAMETERS ###
data_type = ['synthetic', 'lightbox', 'sunlamp'] # Dataset types to process
sample_size = 0.01# Sample size ratios (xxs = 0.05, xs = 0.10, s = 0.25, m = 0.50, l = 1.0)

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
image_size = [400, 640] # Image size to be used (hxw), original size is 1200x1920

zip_output = True # Zip output flag

dataset_name = f'testrt' # Name of the output dataset
### END OF PARAMETERS ###

### CONFIGURATION ###
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Root directory of the project

DATASET_DIR = 'data/src/speedplusv2' # Path to the SPEED+V2 dataset directory

DATASET_PATH = os.path.join(ROOT_PATH, DATASET_DIR)
BLENDER_ENV = 'spv2_2_coco_annotation/blender/3D_model_scene.blend' # Path to the Blender environment file
BE_PATH = os.path.join(ROOT_PATH, BLENDER_ENV)
BLENDER_SCRIPT = 'spv2_2_coco_annotation/blender/coco_annotations.py' # Path to the Blender script for COCO annotations
BS_PATH = os.path.join(ROOT_PATH, BLENDER_SCRIPT)
OUTPUT_DIR = f'data/exp/{dataset_name}' # Path to the output directory for the COCO dataset
OUTPUT_PATH = os.path.join(ROOT_PATH, OUTPUT_DIR)

# Blender parameters
blender_params = {
    'blender_executable': 'blender',
    'blender_file': BE_PATH,
    'blender_script': BS_PATH,
    'render_mask': False,
    'image_size': image_size
}
### END OF CONFIGURATION ###

dataset = DatasetParser(
    root_dir=ROOT_PATH,
    data_dir=DATASET_PATH,
    output_dir=OUTPUT_PATH,
    blender_params=blender_params,
    data_type=data_type,
    sample_size=sample_size,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    test_ratio=test_ratio,
    image_size=image_size,

    zip_output=zip_output,
)
dataset.parse() 
