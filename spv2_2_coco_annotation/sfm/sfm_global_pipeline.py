#!/usr/bin/python
#! -*- encoding: utf-8 -*-
"""
MODIFIED BY: Matthieu Marchal (SII Internship)
LAST UPDATED: 2024-04-04
DESCRIPTION: 
    This script performs the Global Structure from Motion (SfM) pipeline from openMVG on a set of images.
COMMENTS: 
    Please refer to the official openMVG github/documentation for more information.
"""
# This file is part of OpenMVG (Open Multiple View Geometry) C++ library.

# Python implementation of the bash script written by Romuald Perrot
# Created by @vins31
# Modified by Pierre Moulon
#
# this script is for easy use of OpenMVG
#
# usage : python openmvg.py image_dir output_dir
#
# image_dir is the input directory where images are located
# output_dir is where the project must be saved

import os
import subprocess
import sys

# Base directory for data storage
RELATIVE_ROOT = '../..'
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), RELATIVE_ROOT))

# OpenMVG binary directory
OPENMVG_SFM_BIN = os.path.join(ROOT_PATH, "lib/openMVG_Build/Linux-x86_64-RELEASE")

# OpenMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = os.path.join(ROOT_PATH, "lib/openMVG/src/openMVG/exif/sensor_width_database")

if len(sys.argv) < 4:
    print("Usage %s image_dir output_dir f_param" % sys.argv[0])
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]
f_param = sys.argv[3]

matches_dir = os.path.join(output_dir, "matches")
reconstruction_dir = os.path.join(output_dir, "reconstruction_global")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

print("Using input dir  : ", input_dir)
print("      output_dir : ", output_dir)

# Create the output/matches folder if not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

print("1. Intrinsics analysis")
pIntrisics = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListing"), "-i", input_dir, "-o", matches_dir, "-d", camera_file_params, "-f", f_param])
pIntrisics.wait()

print("2. Compute features")
pFeatures = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"), "-i", matches_dir + "/sfm_data.json", "-o", matches_dir, "-m", "SIFT"])
pFeatures.wait()

print("3. Compute matching pairs")
pPairs = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_PairGenerator"), "-i", matches_dir + "/sfm_data.json", "-o", matches_dir + "/pairs.bin"])
pPairs.wait()

print("4. Compute matches")
pMatches = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"), "-i", matches_dir + "/sfm_data.json", "-p", matches_dir + "/pairs.bin", "-o", matches_dir + "/matches.putative.bin", "-r", "0.8"])
pMatches.wait()

print("5. Filter matches")
pFiltering = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"), "-i", matches_dir + "/sfm_data.json", "-m", matches_dir + "/matches.putative.bin", "-g", "e", "-o", matches_dir + "/matches.e.bin"])
pFiltering.wait()

# Create the reconstruction if not present
if not os.path.exists(reconstruction_dir):
    os.mkdir(reconstruction_dir)

print("6. Do Sequential/Incremental reconstruction")
pRecons = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfM"), "--sfm_engine", "STELLAR", "--input_file", matches_dir + "/sfm_data.json", "--match_dir", matches_dir, "--output_dir", reconstruction_dir])
pRecons.wait()

print("7. Colorize Structure")
pRecons = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeSfM_DataColor"), "-i", reconstruction_dir + "/sfm_data.bin", "-o", os.path.join(reconstruction_dir, "colorized.ply")])
pRecons.wait()
