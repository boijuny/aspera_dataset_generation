import os
import argparse
import logging
from PIL import Image
import torch
from torchvision import transforms
from cyclegan_turbo import CycleGAN_Turbo
from my_utils.training_utils import build_transform
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=False, help='the prompt to be used. It is required when loading a custom model_path.')
    parser.add_argument('--model_name', type=str, default=None, help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default=None, help='path to a local model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--image_prep', type=str, default='resize_512x512', help='the image preparation method')
    parser.add_argument('--direction', type=str, default=None, help='the direction of translation. None for pretrained models, a2b or b2a for custom paths.')
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name is None != args.model_path is None:
        raise ValueError('Either model_name or model_path should be provided')

    if args.model_path is not None and args.prompt is None:
        raise ValueError('prompt is required when loading a custom model_path.')

    if args.model_name is not None:
        assert args.prompt is None, 'prompt is not required when loading a pretrained model.'
        assert args.direction is None, 'direction is not required when loading a pretrained model.'

    # initialize the model
    logging.info('Initializing the model...')
    model = CycleGAN_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    logging.info('Model initialized successfully.')

    T_val = build_transform(args.image_prep)

    for folder_name in os.listdir(args.image_dir):
        folder_path = os.path.join(args.image_dir, folder_name)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

        # Adding tqdm progress bar for image processing
        processed_images = 0
        for file_name in tqdm(image_files, desc=f"Processing {folder_name}"):
            file_path = os.path.join(folder_path, file_name)
            logging.info(f'Processing file: {file_path}')
            try:
                input_image = Image.open(file_path).convert('RGB')

                # translate the image
                with torch.no_grad():
                    input_img = T_val(input_image)
                    x_t = transforms.ToTensor()(input_img)
                    x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).cuda()
                    output = model(x_t, direction=args.direction, caption=args.prompt)

                output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
                output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)

                # save the output image
                folder_output = os.path.join(args.output_dir, folder_name)
                os.makedirs(folder_output, exist_ok=True)
                output_pil.save(os.path.join(folder_output, file_name))
                logging.info(f'Successfully processed and saved file: {os.path.join(folder_output, file_name)}')
                processed_images += 1
            except Exception as e:
                logging.error(f'Error processing file {file_path}: {e}')
