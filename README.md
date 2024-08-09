## 📝 Table of Contents

- [About](#-about)
- [Getting Started](#-getting-started)
- [COCO Dataset Format](#-coco-dataset-format)
- [Usage](#-usage)

## 🧐 About
This repository allows you to generate synthetic images, annotated spv2 dataset format into spv2-COCO format, and enhance synthetic images using cycleGAN turbo.
### Components 
This repository contains :
1. `Syntethic_generation\`: Utilizes Blender to generate synthetic images with their pose in SPV2 format.
2. `spv2_2_coco_annotation\`: Converts and annotates SPV2 format dataset into the COCO format for keypoint detection (pose estimation).
3. `domain_adaptation\`: Provide tools to translate images from synthetic domain to "lightbox" or "sunlamp" 
domain. (cycleGAN-turbo)
4. `data\`: Contains both source and created datasets.

**Everything can be runned locally exepted the training and inferencing of the AI-based model (cycleGAN-turbo here). They are performed using AWS (SageMaker and S3), dedicated scripts are provided.**

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will guide you through setting up your local environment to work with the project.

### Prerequisites

Ensure you have the following software installed:

- **Git**: For cloning and managing the repository.
- **Blender**: Required for generating synthetic images and annotating datasets.
- **Python**: The programming language used in the project, along with `pip` for package management.
- **Virtual Environment**: To manage Python dependencies.
- **Git LFS**: To handle large files in the repository.

### Installing

#### Cloning the Repository

1. Open your terminal.
2. Navigate to the directory where you want to clone the project.
3. Clone the repository using Git:

   ```bash
   git clone https://gitlab.sudouest.sii.fr/community/tmd/co-operate/aspera/aspera_synthetic_generation.git
   ```

4. Navigate into the project directory:

   ```bash
   cd aspera_synthetic_generation
   ```

#### Blender

Blender is needed for dataset generation. To install Blender:

1. Visit the [official Blender download page](https://www.blender.org/download/).
2. Download and install the appropriate version for your operating system.
3. Confirm the installation by running:

   ```bash
   blender --version
   ```

#### Python, Pip, and Virtual Environment

Set up Python and its dependencies as follows:

1. Ensure Python is installed. Download it from the [official Python website](https://www.python.org/downloads/).
2. Verify `pip` installation:

   ```bash
   pip --version
   ```

3. Create a virtual environment in your project directory:

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - **Windows**:

     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

5. Install project dependencies from the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

#### Git Large File Storage (Git LFS)

Install and configure Git LFS for managing large files:

1. Follow the [Git LFS installation guide](https://git-lfs.github.com/).
2. Initialize Git LFS in your cloned repository:

   ```bash
   git lfs install
   ```

3. Pull the LFS files to your local repository:

   ```bash
   git lfs pull
   ```

## 🎈 Usage

### Dataset Generation

To generate a 3D dataset using Blender, navigate to the `synthetic_generation` directory and refer to the [spv2_annotation/README.md](synthetic_generation/README.md) file located there.

### SPV2 2 COCO Annotation 

To convert and annotate the SPV2 dataset to the COCO format, navigate to the `spv2_2_coco_annotation` directory and refer to the [spv2_2_coco_annotation/README.md](spv2_2_coco_annotation/README.md) for detailed steps and usage.

### Domain Adaptation

To convert synthetic images into "lightbox" or "sunlamp" domain images, navigate to the `spv2_annotation` directory and refer to the [domain_adaptation/README.md](domain_adaptation/README.md) for detailed steps and usage.

## 📦 Documentation

- **Internship Report**: [Matthieu Marchal's report on ASPERA project (2024)]()
- **Official COCO dataset documentation:** [COCO documentation](https://cocodataset.org/#format-data).
- **CycleGAN-turbo repository:**[GaParmar](https://github.com/GaParmar/img2img-turbo/tree/main?tab=readme-ov-file)

