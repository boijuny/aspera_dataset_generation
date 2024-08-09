import os
import sys
import shutil
import zipfile
import boto3
import logging
import argparse
import subprocess

"""
AUTHOR: Matthieu Marchal (SII Internship)

LAST UPDATED: 2024-08-07

DESCRIPTION:
    This script runs inferences of cyclegan-turbo model
USAGE:
    This script is designed to be used on AWS SageMaker.

CONFIGURATION:
    All configuration parameters are passed as command-line arguments.
"""

logging.basicConfig(level=logging.INFO)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

class PrepareInstance:
    def __init__(self, repo, custom_file='dir_inference_unpaired.py'):
        self.repo = repo
        self.custom_file = custom_file
        self.path = os.path.join('img2img-turbo/src', custom_file)

    def prepare_instance(self):
        logging.info(f'Preparing instance')
        subprocess.run(['git', 'clone', self.repo], check=True)
        logging.info(f'Moving custom inference file')
        shutil.move(self.custom_file, self.path)
        os.chdir('img2img-turbo')
        os.makedirs('ckpt', exist_ok=True)
        logging.info(f'Created {os.path.abspath("data")} and {os.path.abspath("ckpt")}')
        return os.path.abspath('data'), os.path.abspath('ckpt')
        

class PrepareData:
    def __init__(self, base_dir, data_bucket, data_zip, model_ckpt, data_path, ckpt_path):
        self.s3 = boto3.client('s3')
        self.data_bucket = data_bucket
        self.base_dir = base_dir
        self.data_zip =  os.path.basename(data_zip)
        self.data_folder = self.data_zip.split('.')[0]
        self.ckpt_s3 = model_ckpt
        self.ckpt = os.path.basename(model_ckpt)
        self.ckpt_path = os.path.join(ckpt_path,self.ckpt)
        self.data_path = data_path

    def prepare_data(self):
        logging.info(f'Loading data {self.data_zip} at {self.base_dir}')
        zip_file = os.path.join(self.base_dir, self.data_zip)
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(self.data_path)
        os.remove(zip_file)
        logging.info(f'Data has been downloaded and unzipped successfully at {self.data_path}.')
        logging.info(f'Checking {self.data_path} directory: {os.listdir(self.data_path)}')

        logging.info(f'Downloading ckpt {self.ckpt} at {self.ckpt_s3} to {self.ckpt_path} ')
        self.s3.download_file(self.data_bucket, self.ckpt_s3, self.ckpt_path)
        logging.info(f'Downloaded ckpt file at : {self.ckpt_path}')

        logging.info(f'Chekpoints has been downloaded successfully at {self.ckpt_path}.')
        
        dataset_path = os.path.join(self.data_path,self.data_folder)
        images_path = os.path.join(dataset_path,'synthetic/images')
        return dataset_path, images_path, self.ckpt_path
        


class RunModel:
    def __init__(self, model_dir, ckpt_path, data_path, images_type):
        self.ckpt_path = ckpt_path
        self.model_dir = model_dir
        self.data_path = data_path
        self.images_path = os.path.join(data_path,'synthetic/images')
        os.makedirs(os.path.join(data_path,'synthetic/images_gan'),exist_ok=True)
        self.output_dir = os.path.join(data_path,'synthetic/images_gan')
        self.images_type = images_type
        
        
    def run_model(self):
        os.environ['NCCL_P2P_DISABLE'] = '1'
        command = [
            'python', 'src/dir_inference_unpaired.py', '--model_path', f'{self.ckpt_path}', '--image_dir', f'{self.images_path}',
            '--prompt','A 2 B','--direction','a2b','--output_dir',f'{self.output_dir}','--image_prep','no_resize'
        ]
        subprocess.run(command, check=True)
        logging.info(f'Inferences are saved at {self.model_dir}')
        
    def build_output(self):
        shutil.rmtree(self.images_path)
        shutil.move(self.output_dir,self.images_path)
        shutil.move(self.data_path, self.model_dir)
        if self.images_type =='lightbox':
            shutil.rmtree(os.path.join(self.data_path,'sunlamp'))
        elif self.images_type =='sunlamp':
            shutil.rmtree(os.path.join(self.data_path,'lightbox'))

def main(args):
    prepare_instance = PrepareInstance(args.repo)
    data_path,ckpt_path = prepare_instance.prepare_instance()

    prepare_data = PrepareData(args.base_dir,args.data_bucket, args.data_file, args.model_ckpt, data_path,ckpt_path)
    dataset_path, images_path, ckpt_path = prepare_data.prepare_data()
    
    os.listdir(images_path)
    run_model = RunModel(args.model_dir, ckpt_path, dataset_path, images_path)
    run_model.run_model()
    run_model.build_output()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a CycleGAN model on AWS SageMaker')
    parser.add_argument('--repo', type=str, default="https://github.com/GaParmar/img2img-turbo.git", help='Git repository URL containing the training code')
    parser.add_argument('--base-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'], help='Base directory for data processing')
    parser.add_argument('--data-bucket', type=str, required=True, help='Data bucket name')
    parser.add_argument('--data-file', type=str, required=True, help='Data ZIP file name')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'], help='Directory to save trained model')
    parser.add_argument('--model-ckpt', type=str, required=True, help='Model weights')
    parser.add_argument('--images-type',type=str, required=True, help='Type of images in dataset (lightbox or sunlamp)')

    args = parser.parse_args()
    main(args)
