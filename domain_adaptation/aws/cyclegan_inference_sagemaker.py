import sagemaker
from sagemaker.pytorch import PyTorch
import logging

"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-08-07

DESCRIPTION:
    This script launches a SageMaker job to build a dataset composed of enhanced synthetic images with cycleGAN-turbo model.

    ###! WARNING ! : Please carefully use an appropriate INSTANCE_TYPE and INSTANCE_COUNT to avoid UNEXPECTED COSTS.###
    ml.g5.16xlarge for training on GPU = 5$/hour, 
    ml.c5.xlarge for debugging on CPU < 1$/hour.
    
USAGE:
    On an AWS notebook instance, run the script. Ensure the notebook instance has the required files:
        - cyclegan_inference_sagemaker.py
        - src/
            - cyclegan_inference_sagemaker.py
            - cyclegan_inference_sagemaker.py
            - requirements.txt
"""

### CONFIGURATION ###
JOB_NAME = 'cyclegan-inference-sy2su' 

INSTANCE_TYPE = 'ml.g5.16xlarge'
INSTANCE_COUNT = 1

DATA_BUCKET = 'sagemaker-data-aspera'
DATA_ZIP  = 'pose-estimation/spv2-COCO-xs-200x320.zip'
IMAGES_TYPE = 'sunlamp'

OUTPUT_BUCKET = 'sagemaker-output-aspera'
OUTPUT_PREFIX = f'cycleGAN/{JOB_NAME}'

# model
MODEL_CKPT = 'cyclegan/ckpt/sy2su_5001.pkl'


TAGS = [
    {'Key': 'Owner', 'Value': 'matthieu.marchal@sii.fr'},
    {'Key': 'Project', 'Value': 'cyclegan-inference'},
    {'Key': 'CostCenter', 'Value': '25640'},
    {'Key': 'RunningPolicy', 'Value': '24/7'}
]

#####################

# configure the logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Add a stream handler
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(stream_handler)

role = sagemaker.get_execution_role()

hyperparameters = {
    'data-bucket': DATA_BUCKET,
    'data-file': DATA_ZIP,
    'images-type': IMAGES_TYPE,
    'model-ckpt': MODEL_CKPT
}

estimator = PyTorch(entry_point='cyclegan_inference.py',
                    source_dir='src',
                    role=role,
                    instance_count=INSTANCE_COUNT,
                    instance_type=INSTANCE_TYPE,
                    framework_version='2.0.1',
                    py_version='py310',
                    hyperparameters=hyperparameters,
                    output_path=f's3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}',
                    tags=TAGS
                    )

training_input = sagemaker.TrainingInput(s3_data=f's3://{DATA_BUCKET}/{DATA_ZIP}', content_type='application/zip')
estimator.fit({'training': training_input}, job_name=JOB_NAME)
