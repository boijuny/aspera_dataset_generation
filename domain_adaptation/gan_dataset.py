#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AUTHOR: Matthieu Marchal (SII Internship)
LAST MODIFIED: 2024-06-14

DESCRIPTION: This script converts the SPEED+V2 dataset to the COCO format using Blender to render images and annotations.

NOTE: This script is specific to the SPEED+V2 dataset format.
"""

import sys
import os
import shutil
import cv2

# Update the path to include the root project directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### PARAMETERS ###
dataset_name = 'cycleGAN-sy2su-300'
image_number = 300
image_size = [200, 320]  # Image size to be used (height x width)
create_zip = True  # Flag to determine if output should be zipped
### END OF PARAMETERS ###

### CONFIGURATION ###
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_DIR = 'data/src/speedplusv2'
SOURCE_DIR = 'synthetic/images'
TARGET_DIR = 'sunlamp/images'
SOURCE_PATH = os.path.join(ROOT_PATH, DATASET_DIR, SOURCE_DIR)
TARGET_PATH = os.path.join(ROOT_PATH, DATASET_DIR, TARGET_DIR)
OUTPUT_DIR = f'data/exp/{dataset_name}'
OUTPUT_PATH = os.path.join(ROOT_PATH, OUTPUT_DIR)
### END OF CONFIGURATION ###

def zip_output(output_path):
    """Zips the specified directory."""
    shutil.make_archive(output_path, 'zip', output_path)

def write_readme(output_path, dataset_name, image_number, image_size):
    """Generates a README file in the specified directory."""
    file = os.path.join(output_path, 'README.md')
    readme_content = f"""# {dataset_name}

This dataset contains {image_number} images.
The images are {image_size[0]}x{image_size[1]} pixels.
CycleGAN-turbo network provided by [GaParmar](https://github.com/GaParmar/img2img-turbo/tree/main?tab=readme-ov-file)
"""
    
    with open(os.path.join(output_path, 'README.md'), 'w') as file:
        file.write(readme_content)

def process_images(image_path, output_dir, image_limit, size):
    """Processes and moves images from source to target directory."""
    images = os.listdir(image_path)[:image_limit]
    for image in images:
        img = cv2.imread(os.path.join(image_path, image))
        img = cv2.resize(img, (size[1], size[0]))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        cv2.imwrite(os.path.join(output_dir, image), img)

def build_dataset(source_path, target_path, output_dir, image_number, image_size, create_zip):
    """Builds the dataset from source and target directories."""
    train_A = os.path.join(output_dir, 'train_A')
    train_B = os.path.join(output_dir, 'train_B')
    test_A = os.path.join(output_dir, 'test_A')
    test_B = os.path.join(output_dir, 'test_B')

    # Process and move images
    process_images(source_path, train_A, image_number, image_size)
    process_images(target_path, train_B, image_number, image_size)
    process_images(source_path, test_A, image_number, image_size)
    process_images(target_path, test_B, image_number, image_size)

    # Write fixed prompt files
    for prompt_file, content in [('fixed_prompt_a.txt', 'A'), ('fixed_prompt_b.txt', 'B')]:
        with open(os.path.join(output_dir, prompt_file), 'w') as f:
            f.write(content)

    # Optionally zip the output directory
    if create_zip:
        zip_output(output_dir)

if __name__ == '__main__':
    write_readme(OUTPUT_PATH, dataset_name, image_number, image_size)
    build_dataset(SOURCE_PATH, TARGET_PATH, OUTPUT_PATH, image_number, image_size, create_zip)
