#!/usr/bin/env python
"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-04-04

DESCRIPTION:
    This script calculates the percentage of dark pixels in images and provides enhanced visualizations. 
    It allows modifying images by reducing contrast, applying blur, and increasing brightness before copying 
    them to a new folder if their dark pixel percentage is within specified bounds.

USAGE:
    python3 filter_dark_pixels.py

CONFIGURATION:
    All configurations are adjustable in the "CONFIGURATION" section below.

COMMENTS:

MODIFICATIONS:
    For future modifications, adjust parameters in the "CONFIGURATION" section as needed.
"""

import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from typing import List, Tuple
import shutil
from tqdm import tqdm

### CONFIGURATION ###
# Base directory for data storage
RELATIVE_ROOT = '../..'
ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), RELATIVE_ROOT))

# Paths and parameters configuration
SOURCE_FOLDER = 'data/src/speedplusv2/lightbox/images'  # Folder containing images to process
DESTINATION_FOLDER = 'data/src/sfm/lightbox/satellite'  # Folder to save processed images

# Convert paths to absolute paths
SOURCE_PATH = os.path.join(ROOT_PATH, SOURCE_FOLDER)
DESTINATION_PATH = os.path.join(ROOT_PATH, DESTINATION_FOLDER)

# Parameters for image processing
RESIZE_FACTOR = 0.2  # Factor to resize the image before applying enhancements
BLUR_RADIUS = 2  # Radius of the Gaussian Blur to apply
CONTRAST_FACTOR = 2  # Factor to enhance contrast
DARKNESS_THRESHOLD = 35  # Threshold for considering a pixel "dark"

# Parameters for filtering and copying images
COPY_IMAGES = False  # Whether to copy images that meet dark pixel criteria
DARKNESS_LB = 80  # Lower bound of dark pixel percentage for copying
DARKNESS_HB = 90  # Upper bound of dark pixel percentage for copying
SHOW_PLOT = True  # Whether to display plots

### END OF CONFIGURATION ###

def calculateDarkPixelPercentage(image_array: np.ndarray, DARKNESS_THRESHOLD: int) -> float:
    """
    Calculate the percentage of dark pixels in an image.
    Args:
        image_array: The numpy array of the image.
        DARKNESS_THRESHOLD: The threshold value below which a pixel is considered dark.
    Returns:
        The percentage of dark pixels in the image.
    """
    dark_pixels = np.sum(image_array < DARKNESS_THRESHOLD)
    total_pixels = image_array.size
    return (dark_pixels / total_pixels) * 100

def enhanceImage(img: Image, RESIZE_FACTOR: float, BLUR_RADIUS: int, CONTRAST_FACTOR: float) -> Image:
    """
    Apply a Gaussian Blur filter and adjust contrast of an image to reduce graininess and enhance its features.
    
    Args:
    - img: The input PIL Image object.
    - RESIZE_FACTOR: Factor to resize the image by before applying enhancements. 
                     Less than 1 to reduce size, 1 to keep size unchanged.
    - BLUR_RADIUS: The radius of the Gaussian Blur to apply.
    - CONTRAST_FACTOR: Factor by which the image contrast will be increased. 
                       A value of 1 will leave the contrast unchanged, while values greater than 1 will increase contrast.
    
    Returns:
    - The enhanced PIL Image object.
    """
    # Optionally resize the image to reduce the noise/grain
    if RESIZE_FACTOR != 1:
        new_size = (int(img.width * RESIZE_FACTOR), int(img.height * RESIZE_FACTOR))
        img = img.resize(new_size)
    
    # Apply Gaussian Blur to smooth the image
    img = img.filter(ImageFilter.BoxBlur(BLUR_RADIUS))
    
    # Adjust the contrast of the image
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(CONTRAST_FACTOR)
    
    return img
def processImagesInFolder(source_path: str, RESIZE_FACTOR: float, BLUR_RADIUS: int,CONTRAST_FACTOR: float, DARKNESS_THRESHOLD: int) -> List[Tuple[str, float]]:
    file_data = []
    files = os.listdir(source_path)
    print(f"Processing images in {source_path}...")
    with tqdm(total=len(files), desc="Processing") as pbar:
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                try:
                    image_path = os.path.join(source_path, file_name)
                    with Image.open(image_path) as img:
                        gray_image = img.convert('L')
                        smooth_gray_image = enhanceImage(gray_image, RESIZE_FACTOR, BLUR_RADIUS, CONTRAST_FACTOR)
                        dark_pixel_percentage = calculateDarkPixelPercentage(np.array(smooth_gray_image), DARKNESS_THRESHOLD)
                        file_data.append((file_name, dark_pixel_percentage))
                except IOError:
                    print(f"Cannot open {file_name}.")
            pbar.update(1)
    return file_data

def enhancedVisualization(file_data: List[Tuple[str, float]], source_path: str, intervals: List[Tuple[int, int]] = [(0, 10), (10,20), (20, 30), (30,40), (40, 50), (50,60), (60, 70), (70,80), (80, 90), (90, 100)]):
    """
    Provides an enhanced visualization of the dark pixel percentage distribution and displays sample images for specified intervals.
    """
    # Plotting the histogram of dark pixel percentages
    dark_percentages = [darkness for _, darkness in file_data]
    plt.figure(figsize=(10, 6))
    plt.hist(dark_percentages, bins=range(0, 105, 5), color='blue', alpha=0.7)
    plt.title('Histogram of Dark Pixel Percentages in Images')
    plt.xlabel('Dark Pixel Percentage (%)')
    plt.ylabel('Number of Images')
    plt.grid(True)
    if SHOW_PLOT:
        plt.show()

    # Displaying sample images for each specified interval
    for start, end in intervals:
        sample_files = [file_name for file_name, dark_percentage in file_data if start <= dark_percentage < end]
        if sample_files:
            # display random sample image from the interval
            sample_image_name = os.path.join(source_path, np.random.choice(sample_files))
            img = Image.open(sample_image_name)
            plt.figure()
            plt.imshow(img)
            plt.title(f'Sample Image: {start}-{end}% Dark Pixels')
            plt.axis('off')  # Hide axis labels
            if SHOW_PLOT:
                plt.show()

def modifyAndCopyImages(file_data: List[Tuple[str, float]], source_path: str, destination_path: str,
                           RESIZE_FACTOR: float, BLUR_RADIUS: int,
                           CONTRAST_FACTOR: float, brightness_factor: float):
    """
    Copy images from the source folder to the destination folder with modifications:
    reduced contrast, blurring, and increased brightness.

    Args:
    - source_path: Path to the source folder containing images.
    - destination_path: Path to the destination folder where modified images will be saved.
    - RESIZE_FACTOR: Factor to resize the image by before applying enhancements.
    - BLUR_RADIUS: The radius of the Gaussian Blur to apply for blurring.
    - CONTRAST_FACTOR: Factor by which the image contrast will be decreased.
    - brightness_factor: Factor by which the image brightness will be increased.

    Each image from the source folder will be opened, modified, and saved in the destination folder
    with the same filename.
    """
    os.makedirs(destination_path, exist_ok=True)  # Ensure destination directory exists
    c = 0
    print(f"Selecting and Copying images to {destination_path}...")
    with tqdm(total=len(file_data), desc="Selecting and copying") as pbar:
        for file_name, dark_percentage in file_data:
            if DARKNESS_LB < dark_percentage < DARKNESS_HB:
                source_path = os.path.join(source_path, file_name)
                save_path = os.path.join(destination_path, file_name)
                c+=1
                try:
                    # Open source image
                    with Image.open(source_path) as img:
                        # Apply Gaussian Blur
                        img = img.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))

                        # Decrease contrast
                        contrast = ImageEnhance.Contrast(img)
                        img = contrast.enhance(CONTRAST_FACTOR)

                        # Increase brightness
                        brightness = ImageEnhance.Brightness(img)
                        img = brightness.enhance(brightness_factor)

                        # Save modified image to destination folder
                        img.save(save_path)

                except IOError as e:
                    print(f"Error processing {file_name}: {e}")
            pbar.update(1)
    print(f"Saved {c} images.")

def main():
    # Convert relative paths to absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(script_dir, SOURCE_PATH)
    destination_path = os.path.join(script_dir, SOURCE_PATH)

    # Call functions as before but now with the paths being dynamically constructed
    file_data = processImagesInFolder(source_path, RESIZE_FACTOR, BLUR_RADIUS, CONTRAST_FACTOR, DARKNESS_THRESHOLD)
    enhancedVisualization(file_data, source_path)

    if COPY_IMAGES:
        modifyAndCopyImages(file_data, source_path, destination_path, RESIZE_FACTOR, BLUR_RADIUS, CONTRAST_FACTOR, 1)  # Adjusted parameters as necessary

    if SHOW_PLOT:
        plt.show()

if __name__ == "__main__":
    main()
