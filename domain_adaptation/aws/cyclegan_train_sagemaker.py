import sagemaker
from sagemaker.pytorch import PyTorch
import logging

"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-07-14

DESCRIPTION:
    This script launches a SageMaker training job for CycleGAN-Turbo using the PyTorch estimator.

    ###! WARNING ! : Please carefully use an appropriate INSTANCE_TYPE and INSTANCE_COUNT to avoid UNEXPECTED COSTS.###
    ml.g5.16xlarge for training on GPU = 5$/hour, 
    ml.c5.xlarge for debugging on CPU < 1$/hour.
    
USAGE:
    On an AWS notebook instance, run the script. Ensure the notebook instance has the required files:
        - train_cyclegan_sagemaker.py
        - src/train_cyclegan_turbo.py
        - src/requirements.txt

CONFIGURATION:
    - data_bucket: S3 bucket containing the training data. (str)
    - data_zip: Name of the zip file containing the training data. (str)
    - output_bucket: S3 bucket where the training output will be stored. (str)
    - job_name: Unique name of the training job. (str)
    - output_prefix: Prefix of the output path. (str)
"""

### CONFIGURATION ###
INSTANCE_TYPE = 'ml.g5.16xlarge'
INSTANCE_COUNT = 1

DATA_BUCKET = 'sagemaker-data-aspera'
DATA_ZIP  = 'cyclegan/cycleGAN-sy2su-300.zip'

OUTPUT_BUCKET = 'sagemaker-output-aspera'
JOB_NAME = 'cyclegan-turbo-training-sy2su-300' 
OUTPUT_PREFIX = f'cycleGAN/{JOB_NAME}'

WANDB_API_KEY = '' # ENTER you wandb API key
# Hyperparameters for CycleGAN-Turbo
LEARNING_RATE = "1e-6"
MAX_TRAIN_STEPS = 5000
TRAIN_BATCH_SIZE = 1
IMG_PREP = "no_resize"
GRADIENT_ACCUMULATION_STEPS = 1
REPORT_TO = 'wandb'
TRACKER_PROJECT_NAME = 'AWS-CycleGAN-Turbo-sy2su'
VALIDATION_STEPS = 250
LAMBDA_GAN = 0.5
LAMBDA_IDT = 1
LAMBDA_CYCLE = 1

TAGS = [
    {'Key': 'Owner', 'Value': 'matthieu.marchal@sii.fr'},
    {'Key': 'Project', 'Value': 'CycleGAN-Turbo-Training'},
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
    'data-file': DATA_ZIP,
    'learning-rate': LEARNING_RATE,
    'max-train-steps': MAX_TRAIN_STEPS,
    'train-batch-size': TRAIN_BATCH_SIZE,
    'train-img-prep': IMG_PREP,
    'val-img-prep': IMG_PREP,
    'gradient-accumulation-steps': GRADIENT_ACCUMULATION_STEPS,
    'report-to': REPORT_TO,
    'tracker-project-name': TRACKER_PROJECT_NAME,
    'validation-steps': VALIDATION_STEPS,
    'lambda-gan': LAMBDA_GAN,
    'lambda-idt': LAMBDA_IDT,
    'lambda-cycle': LAMBDA_CYCLE
}

estimator = PyTorch(entry_point='cyclegan_train.py',
                    source_dir='src',
                    role=role,
                    instance_count=INSTANCE_COUNT,
                    instance_type=INSTANCE_TYPE,
                    framework_version='2.0.1',
                    py_version='py310',
                    hyperparameters=hyperparameters,
                    output_path=f's3://{OUTPUT_BUCKET}/{OUTPUT_PREFIX}',
                    environment={'WANDB_API_KEY': WANDB_API_KEY},
                    tags=TAGS
                    )

training_input = sagemaker.TrainingInput(s3_data=f's3://{DATA_BUCKET}/{DATA_ZIP}', content_type='application/zip')
estimator.fit({'training': training_input}, job_name=JOB_NAME)
