## Table of Contents
- [Overview](#overview)
- [Dataset](#cyclegan-turbo-dataset-format)
- [AWS](#training-and-inferencing-on-aws)

## Overview

The `domain_adaptation` module performs domain translation from synthetic images to 'lightbox' or 'sunlamp' images domain. It is based on the training of cycleGAN-turbo network provided by [GaParmar](https://github.com/GaParmar/img2img-turbo/tree/main?tab=readme-ov-file). 
### Components

In the section, you can find :
- `gan_dataset.py`: a script that builds a dataset for cycleGAN training.
- The scripts to use for AWS model training and model inference to build the translated datasets.
- The weights of each synthetic->lightbox and synthetic->sunlamp trained models.

## cycleGAN-turbo dataset format 

```bash
  cycleGAN-sy2xx-/
  ├── README.md
  ├── fixed_prompt_a.txt
  ├── fixed_prompt_b.txt
  ├── train_A/
  ├── test_A/
  ├── train_B/
  ├── test_B/
      ├── img00001.jpg
      └── ...
```

## Training and Inferencing on AWS

 The `aws` folder provides materials too either train a cycleGAN-turbo model or run inferences on spv2-COCO dataset (convert synthetic to sunlamp for exemple).

 To have more details on how to train or run inferences using AWS SageMaker and S3 please refer to the [README_AWS]().

## Authors
- [Matthieu Marchal](https://github.com/boijuny)
- [GaParmar](https://github.com/GaParmar/img2img-turbo/tree/main?tab=readme-ov-file)
