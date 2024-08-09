import sys
import os
import shutil
import random
from tqdm import tqdm
import subprocess
import json
import cv2
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def loadJson(json_path):
    """
    Loads a JSON file.

    Parameters:
    - json_path: Path to the JSON file.

    Returns:
    - data: The JSON data.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: The file at {json_path} was not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Failed to decode the JSON file.")
        return None

def saveJson(data, json_path):
    """
    Saves a JSON file.

    Parameters:
    - data: The JSON data.
    - json_path: Path to the JSON file.
    """
    try:
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            print(f"Step Completed: JSON data saved to {json_path}.")
    except Exception as e:
        print(f"Error: {e}")

class DatasetParser:
    """
    A class to parse and process datasets.

    Attributes:
    - root_dir: The root directory.
    - data_dir: The path to the dataset.
    - data_type: The type of data to parse.
    - output_dir: The output directory.
    - blender_params: The parameters for Blender processing.
    - sample_size: The sample size to use. Default: 0.25.
    - train_ratio: The ratio of training data. Default: 0.7.
    - val_ratio: The ratio of validation data. Default: 0.15.
    - test_ratio: The ratio of test data. Default: 0.15.
    - image_size: The size of the images. Default: [200, 640].
    - zip_output: Whether to zip the output directory. Default: True.
    """

    def __init__(self, root_dir, data_dir, output_dir, blender_params, data_type, sample_size=0.25, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, image_size=[200, 320], zip_output=False):
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.data_type = data_type
        self.output_dir = output_dir
        self.blender_params = blender_params

        self.sz = sample_size
        self.trr = train_ratio
        self.var = val_ratio
        self.tsr = test_ratio
        self.isz = image_size
        self.zo = zip_output

        os.makedirs(self.output_dir, exist_ok=True)
        print(f"=== Initializing DatasetParser ===")
        print(f"Root Directory: {self.root_dir}")
        print(f"Data Directory: {self.data_dir}")
        print(f"Data Type: {self.data_type}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Sample Size: {self.sz}")
        print(f"Train Ratio: {self.trr}")
        print(f"Validation Ratio: {self.var}")
        print(f"Test Ratio: {self.tsr}")
        print(f"Image Size: {self.isz}")
        print(f"Zip Output: {self.zo}")
        print(f"=== Initialization Complete ===\n")

    def combineData(self, json_files, sort_key=None):
        """
        Combines the data from multiple JSON files.
        
        Parameters:
        - json_files: The JSON files to combine. Format: [[], ...]
        - sort_key: The key to sort the data by. Default: None.
        """
        combined_data = []
        for json_file in json_files:
            combined_data.extend(json_file)
        if sort_key:
            combined_data.sort(key=lambda x: x[sort_key])
        return combined_data

    def splitCocoData(self, data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Splits the data into training, validation, and test sets.
        
        Parameters:
        - data: The combined dataset.
        - train_ratio: The ratio of training data.
        - val_ratio: The ratio of validation data.
        - test_ratio: The ratio of test data.

        Returns:
        - train_data: Training dataset.
        - val_data: Validation dataset.
        - test_data: Testing dataset.
        """
        if train_ratio + val_ratio + test_ratio != 1:
            raise ValueError("Error: The sum of ratios must be equal to 1.")

        images = data['images']
        num_images = len(images)
        train_end = int(train_ratio * num_images)
        val_end = train_end + int(val_ratio * num_images)

        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]

        def filter_annotations(images_set):
            img_ids = {img['id'] for img in images_set}
            return [ann for ann in data['annotations'] if ann['image_id'] in img_ids]

        train_annotations = filter_annotations(train_images)
        val_annotations = filter_annotations(val_images)
        test_annotations = filter_annotations(test_images)

        def construct_subset(images, annotations):
            return {
                'info': data.get('info', {}),
                'licenses': data.get('licenses', []),
                'images': images,
                'annotations': annotations,
                'categories': data.get('categories', [])
            }

        train_data = construct_subset(train_images, train_annotations)
        val_data = construct_subset(val_images, val_annotations)
        test_data = construct_subset(test_images, test_annotations)

        print(f"Data Split Complete:")
        print(f"  - Total Images: {num_images}")
        print(f"  - Train Images: {len(train_images)}, Annotations: {len(train_annotations)}")
        print(f"  - Validation Images: {len(val_images)}, Annotations: {len(val_annotations)}")
        print(f"  - Test Images: {len(test_images)}, Annotations: {len(test_annotations)}")

        return train_data, val_data, test_data

    def processImgs(self, json_data, image_path, output_path, format='coco'):
        """
        Processes the images by resizing them and saving them in the output folder.

        Parameters:
        - json_data: The JSON data with image paths.
        - image_path: The path to the image folder.
        - output_path: The path to the output folder.
        - format: The format of the JSON file. Default: 'coco'.
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filenaming = 'filename'
        if format == 'coco':
            json_data = json_data['images']
            filenaming = 'file_name'

        with tqdm(total=len(json_data), desc="Processing Images") as pbar:
            for i in range(len(json_data)):
                image_name = json_data[i][filenaming]
                image = cv2.imread(os.path.join(image_path, image_name))
                image = cv2.resize(image, (self.isz[1], self.isz[0]))
                cv2.imwrite(os.path.join(output_path, image_name), image)
                pbar.update(1)

    def processCameraPrm(self):
        """
        Adjusts the camera parameters for resized images based on the new image dimensions.
        """
        print("=== Adjusting Camera Parameters ===")
        camera_json_path = os.path.join(self.data_dir, 'camera.json')
        camera_output_path = os.path.join(self.output_dir, 'camera.json')
        new_height = self.isz[0]
        new_width = self.isz[1]

        camera_params = loadJson(camera_json_path)
        
        # Extract original image dimensions
        original_width = camera_params['Nu']
        original_height = camera_params['Nv']
        
        # Calculate the scale factors for width and height
        scale_width = new_width / original_width
        scale_height = new_height / original_height
        
        # Adjust the focal lengths (fx, fy)
        new_fx = camera_params['fx'] * scale_width
        new_fy = camera_params['fy'] * scale_height
        
        # Adjust the principal points (ccx, ccy)
        new_ccx = camera_params['ccx'] * scale_width
        new_ccy = camera_params['ccy'] * scale_height
        
        # Adjust the camera matrix
        new_camera_matrix = [
            [camera_params['cameraMatrix'][0][0] * scale_width, 0, new_ccx],
            [0, camera_params['cameraMatrix'][1][1] * scale_height, new_ccy],
            [0, 0, 1]
        ]
        
        # Prepare the new camera parameters dictionary
        new_camera_params = {
            "Nu": new_width,
            "Nv": new_height,
            "ppx": camera_params["ppx"],
            "ppy": camera_params["ppy"],
            "fx": new_fx,
            "fy": new_fy,
            "ccx": new_ccx,
            "ccy": new_ccy,
            "cameraMatrix": new_camera_matrix,
            "distCoeffs": camera_params["distCoeffs"]
        }
        saveJson(new_camera_params, camera_output_path)
        print("Camera parameters adjusted and saved.")

    def processKeypointsPrm(self):
        """
        Copies the keypoints parameters to the output directory.
        """
        print("=== Copying Keypoints Parameters ===")
        keypoints_json_path = os.path.join(self.root_dir, 'spv2_annotation/blender/keypoints.json')
        keypoints_output_path = os.path.join(self.output_dir, 'keypoints.json')
        keypoints_params = loadJson(keypoints_json_path)
        saveJson(keypoints_params, keypoints_output_path)
        print("Keypoints parameters copied.")

    def processPrms(self):
        """
        Processes the camera and keypoints parameters.
        """
        print("=== Processing Parameters ===")
        self.processCameraPrm()
        self.processKeypointsPrm()
        # Generate a README.md file
        with open(os.path.join(self.output_dir, 'README.md'), 'w') as f:
            f.write(f'This directory contains the processed dataset.\n')
            f.write(f'## Camera Parameters\n')
            f.write(f'Camera parameters have been adjusted for the resized images: {self.isz[0]}x{self.isz[1]} px.\n')
            f.write(f'## Image Parameters\n')
            f.write(f'Images have been resized from 1200x1920 to {self.isz[0]}x{self.isz[1]} px.\n')
            f.write(f'## Keypoints Parameters\n')
            f.write(f'Expressed in pixels [px].\n')
        print("Parameters processed and README.md generated.")

    def zipOutput(self):
        """
        Zips the output directory.
        """
        print("=== Zipping Output Directory ===")
        zipf = zipfile.ZipFile(f'{self.output_dir}.zip', 'w', zipfile.ZIP_DEFLATED)
        for root, dirs, files in os.walk(self.output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(self.output_dir, '..')))
        zipf.close()
        print("Output directory zipped.")
        shutil.rmtree(self.output_dir)
    
    def parse(self):
        """
        Parses the dataset.
        """
        print("=== Step 1: Parsing Dataset ===")
        for dt in self.data_type:
            print(f"--- Processing Data Type: {dt} ---")
            path = os.path.join(self.data_dir, dt)
            files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]
            data = [loadJson(file) for file in files]
            data = self.combineData(data)
            data = random.sample(data, int(self.sz * len(data)))

            itmp_file = os.path.join(self.output_dir, 'itmp.json')
            otmp_file = os.path.join(self.output_dir, 'otmp.json')

            saveJson(data, itmp_file)
            print(f"Number of Data Samples: {len(data)}")

            # Blender processing
            print(f"--- Step 2: Blender Processing for {dt} Data ---")
            exec_path = self.blender_params['blender_executable']
            env_file = self.blender_params['blender_file']
            script_file = self.blender_params['blender_script']
            args = [
                '--root', self.root_dir,
                '--input-file', itmp_file,
                '--output-file', otmp_file,
                '--image-height', str(self.isz[0]),
                '--image-width', str(self.isz[1])
                ]
            command = [exec_path, '-b', env_file, '-P', script_file, '--'] + args
            print(f"Running Blender Command: {command}")
            subprocess.run(command)
            print("Blender Processing Completed.")
            print('-' * 60)

            data = loadJson(otmp_file)

            if dt == 'synthetic':
                train, validation, test = self.splitCocoData(data, train_ratio=self.trr, val_ratio=self.var, test_ratio=self.tsr)
                output_types = ['train', 'validation', 'test']
            else:
                test = data
                output_types = ['test']

            for output_type in output_types:
                if output_type == 'train':
                    out_data = train
                elif output_type == 'validation':
                    out_data = validation
                elif output_type == 'test':
                    out_data = test

                save_path = os.path.join(self.output_dir, f'{dt}/annotations')
                os.makedirs(save_path, exist_ok=True)
                saveJson(out_data, os.path.join(save_path, f'{output_type}.json'))

                os.makedirs(os.path.join(self.output_dir, f'{dt}/images/{output_type}'), exist_ok=True)
                print(f"Processing Images for {dt}/{output_type}...")
                image_path = os.path.join(path, 'images')
                output_path = os.path.join(self.output_dir, f'{dt}/images/{output_type}')
                self.processImgs(out_data, image_path, output_path)
        
        os.remove(itmp_file)
        os.remove(otmp_file)
        print("=== Dataset Parsing Completed===")
        self.processPrms()  # Process camera and keypoints parameters
        if self.zo:
            self.zipOutput()
