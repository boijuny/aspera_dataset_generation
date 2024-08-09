## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
  - [Converting SPEED+V2 Dataset to COCO Format](#converting-speedv2-dataset-to-coco-format)
  - [Utilities](#utilities)
    - [Data Parsing Utilities](#data-parsing-utilities)
- [COCO Dataset Format](#coco-dataset-format)
  - [Key Components of COCO Format](#key-components-of-coco-format)
- [Authors](#authors)

## Overview

The `spv2_2_coco_annotation` module automates the process of converting the SPEED+V2 synthetic format dataset into the COCO format. This includes:
- Reproducting images poses in Blender with Tango 3D model to generate the annotations (2D keypoints, visibility...)
- Annotating these images with keypoints, bounding boxes, and other required information as per the COCO format.
### Dataset conversion

The annotation and conversion of SPEED+V2 dataset into COCO format is performed with `spv2_2_coco.py` script. Please refer to [Usage](##Usage) section for more details.
### 3D model Reconstruction (sfm)
In `sfm/` you can find scripts that were use to reconstruct the 3D model mesh of Tango satellite using MVG/MVS strategies. Unfortunately theses methods were not good enough so the 3D mesh in this project comes from **lava1302 team's work** [available here](https://github.com/willer94/lava1302).
### Blender scene
In `blender/` you can find all the scenes,scripts and meshes relevant to the dataset annotation task.
## Datasets
### SPEED+V2 dataset format 

```bash
  speedplusv2/
  ├── README.md
  ├── camera.json
  ├── lightbox/...
  ├── sunlamp/...
  └── synthetic/
      ├── images/
      │   ├── img000001.jpg
      │   ├── img000002.jpg
      │   └── ...
      ├── train.json
      └── validation.json
```
### spv2-COCO dataset format

```bash
  spv2-COCO/
  ├── README.md
  ├── camera.json
  ├── keypoints.json
  ├── lightbox/...
  ├── sunlamp/...
  └── synthetic/
      ├── images/
      │   ├── train/
      │   ├── validation/
      │   └── test/
      └── annotations/
          ├── train.json  in COCO format
          ├── validation.json in COCO format
          └── tests.json in COCO format
```
The COCO (Common Objects in Context) format is a standardized format for datasets used in object detection, segmentation, and keypoint detection tasks. Annotations are represented by a JSON file :  

#### Key Components of COCO Format

1. **Images**: Metadata for each image, including ID, width, height, and file name.
2. **Annotations**: Details about objects in images, including bounding boxes, segmentation, and keypoints.
3. **Categories**: Defines the class labels of the objects.

For more details, refer to the [official COCO dataset documentation](https://cocodataset.org/#format-data).

## Setup

Please refer to the [main README](../README.md) for more setup details.

## Usage

### Converting SPEED+V2 Dataset to COCO Format

After setting up the project, to convert the SPEED+V2 dataset to the COCO format, you need to download the SPEED+V2 official dataset [available here](https://zenodo.org/records/5588480). 

  1. Unzip the speedplusv2.zip file
  2. Place it in `data/src/`

Then you can run the `spv2_2_coco.py` script. 

```bash
python spv2_to_coco.py
```

The script will process the dataset according to the parameters defined within it, render the images, and generate the COCO annotations. It has many customizable parameters : 

  ```bash 
    - data_types # ['synthetic', 'lightbox', 'sunlamp']
    - sample_size # 0.10 is considered as xs for exemple
    - train_ratio #0.7 common value
    - val_ratio # 0.15 common value
    - test_ratio # 0.15 common value
    - image_size # [200,320] HxW

    - zip_output # zip output flag
    - other path configurations...
  ```

### Utilities

#### Data Parsing Utilities

The core functionalities for data parsing and processing are encapsulated in the `utils/data_parser.py` script. This script contains the `DatasetParser` class, which is designed to manage the loading, parsing, sampling, and saving of datasets.

##### Class Description

The `DatasetParser` class provides methods for:
- Loading JSON data.
- Validating and combining data entries.
- Sampling data for balanced dataset creation.
- Executing external processes for data transformation (e.g., Blender).
- Splitting data into training, validation, and test sets.
- Saving processed data back to disk.

It is used extensively in the `spv2_to_coco.py` script to convert the SPEED+V2 dataset into the COCO format.

## Acknowledgments 
- [SPEC2021 Challenge](https://kelvins.esa.int/pose-estimation-2021/home/)
- [lava1302 team](https://github.com/willer94/lava1302) - Winner of  SPEC2021 challenge on 'lightbox' dataset.

## Authors
[Matthieu Marchal](https://github.com/boijuny)
