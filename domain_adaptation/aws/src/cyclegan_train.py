import os
import sys
import shutil
import zipfile
import logging
import argparse
import subprocess

"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-08-07

DESCRIPTION:
    This script trains a cyclegan-turbo model.
USAGE:
    This script is designed to be used on AWS SageMaker.

CONFIGURATION:
    All configuration parameters are passed as command-line arguments.
"""

logging.basicConfig(level=logging.INFO)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

class PrepareInstance:
    def __init__(self, repo):
        self.repo = repo

    def prepare_instance(self):
        subprocess.run(['git', 'clone', self.repo], check=True)
        os.chdir('img2img-turbo')
        return os.path.abspath('data')

class PrepareData:
    def __init__(self, base_dir, data_zip, path):
        self.base_dir = base_dir
        self.data_zip = os.path.basename(data_zip)
        self.data_folder = self.data_zip.split('.')[0]
        self.path = path

    def prepare_data(self):
        logging.info(f'Loading data {self.data_zip} at {self.base_dir}')
        zip_file = os.path.join(self.base_dir, self.data_zip)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.path)
        os.remove(zip_file)
        logging.info(f'Data has been downloaded and unzipped successfully at {self.path}.')
        logging.info(f'Checking {self.data_folder} directory: {os.listdir(self.path)}')

class TrainModel:
    def __init__(self, model_dir, data_dir, train_img_prep, val_img_prep, learning_rate, max_train_steps, train_batch_size, gradient_accumulation_steps, report_to, tracker_project_name, validation_steps, lambda_gan, lambda_idt, lambda_cycle):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.train_img_prep = train_img_prep
        self.val_img_prep = val_img_prep
        self.learning_rate = learning_rate
        self.max_train_steps = max_train_steps
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.report_to = report_to
        self.tracker_project_name = tracker_project_name
        self.validation_steps = validation_steps
        self.lambda_gan = lambda_gan
        self.lambda_idt = lambda_idt
        self.lambda_cycle = lambda_cycle

    def train_model(self):
        os.environ['NCCL_P2P_DISABLE'] = '1'
        command = [
            'accelerate', 'launch', '--main_process_port', '29501', 'src/train_cyclegan_turbo.py',
            '--pretrained_model_name_or_path=stabilityai/sd-turbo',
            f'--output_dir={self.model_dir}',
            f'--dataset_folder={self.data_dir}', f'--max_train_steps={self.max_train_steps}',
            f'--train_img_prep={self.train_img_prep}', f'--val_img_prep={self.val_img_prep}',
            f'--train_batch_size={self.train_batch_size}', f'--gradient_accumulation_steps={self.gradient_accumulation_steps}',
            f'--report_to={self.report_to}', f'--tracker_project_name={self.tracker_project_name}',
            '--enable_xformers_memory_efficient_attention', f'--validation_steps={self.validation_steps}',
            f'--lambda_gan={self.lambda_gan}', f'--lambda_idt={self.lambda_idt}', f'--lambda_cycle={self.lambda_cycle}'
        ]
        subprocess.run(command, check=True)
        logging.info(f'Model is saved at {self.model_dir}')

def main(args):
    prepare_instance = PrepareInstance(args.repo)
    data_path = prepare_instance.prepare_instance()
    
    prepare_data = PrepareData(args.base_dir, args.data_file, data_path)
    prepare_data.prepare_data()
    
    train_model = TrainModel(args.model_dir, data_path, args.train_img_prep, args.val_img_prep, args.learning_rate, args.max_train_steps, args.train_batch_size, args.gradient_accumulation_steps, args.report_to, args.tracker_project_name, args.validation_steps, args.lambda_gan, args.lambda_idt, args.lambda_cycle)
    train_model.train_model()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CycleGAN model on AWS SageMaker')
    parser.add_argument('--repo', type=str, default="https://github.com/GaParmar/img2img-turbo.git", help='Git repository URL containing the training code')
    parser.add_argument('--base-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'], help='Base directory for data processing')
    parser.add_argument('--data-file', type=str, required=True, help='Data ZIP file name')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'], help='Directory to save trained model')
    parser.add_argument('--train-img-prep', type=str, required=True, help='Image preprocessing settings for training')
    parser.add_argument('--val-img-prep', type=str, required=True, help='Image preprocessing settings for validation')
    parser.add_argument('--learning-rate', type=str, required=True, help='Learning rate for training')
    parser.add_argument('--max-train-steps', type=int, required=True, help='Maximum number of training steps')
    parser.add_argument('--train-batch-size', type=int, required=True, help='Training batch size')
    parser.add_argument('--gradient-accumulation-steps', type=int, required=True, help='Number of gradient accumulation steps')
    parser.add_argument('--report-to', type=str, required=True, help='Reporting target (e.g., "none", "tensorboard")')
    parser.add_argument('--tracker-project-name', type=str, required=True, help='Project name for tracking experiments')
    parser.add_argument('--validation-steps', type=int, required=True, help='Number of steps between validations')
    parser.add_argument('--lambda-gan', type=float, required=True, help='Lambda weight for GAN loss')
    parser.add_argument('--lambda-idt', type=float, required=True, help='Lambda weight for identity loss')
    parser.add_argument('--lambda-cycle', type=float, required=True, help='Lambda weight for cycle-consistency loss')

    args = parser.parse_args()
    main(args)
