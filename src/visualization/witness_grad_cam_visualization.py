import click
import os
from pathlib import Path
from datetime import date

from omegaconf import open_dict    
import torch
import numpy as np
import cv2
import pandas as pd
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from src.visualization.witness_grad_cam.witness_cam import WitnessGradCAM, \
    WitnessScoreCAM, \
    WitnessGradCAMPlusPlus, \
    WitnessAblationCAM, \
    WitnessXGradCAM, \
    WitnessEigenCAM, \
    WitnessEigenGradCAM, \
    WitnessLayerCAM, \
    WitnessFullGrad

from src.utils.dynamic_import import import_evaluator, import_network_pl
from src.utils.utils import get_cfg_from_mlflow, handle_cfg_compatibility
from src.models.datamodule.datamodule_pl import broadcast_to_3_chs_album, center_normalization_func, ToTensorV2, A
import sys
sys.path.append('./pytorch-grad-cam')
from pytorch_grad_cam.utils.image import show_cam_on_image

IMAGE_SAVE_DIR = 'reports/figures/grad'
METHODS = \
    {"gradcam": WitnessGradCAM,
        "scorecam": WitnessGradCAM,
        "gradcam++": WitnessGradCAM,
        "ablationcam": WitnessGradCAM,
        "xgradcam": WitnessGradCAM,
        "eigencam": WitnessGradCAM,
        "eigengradcam": WitnessGradCAM,
        "layercam": WitnessGradCAM,
        "fullgrad": WitnessGradCAM}

@click.command()
@click.option('-i', 'run_id', required=True, type=str)
@click.option('-m', 'method', default='gradcam', type=str)
@click.option('-b', 'batch_size', default=32, type=int)
@click.option('--gpu', is_flag=True)
@click.option('--aug-smooth', is_flag=True)
@click.option('--eigen-smooth', is_flag=True)
def main(run_id, method, batch_size, gpu, aug_smooth, eigen_smooth):
    print(run_id)



    project_path = Path('.')
    try:         
        target_dir_path = list((project_path / 'model_weights').glob(f'*{run_id}*'))[-1]
        target_dir = target_dir_path.name
        figure_dir = os.path.join(project_path, 'reports/figures', target_dir)
        if not os.path.exists(figure_dir):
            os.mkdir(figure_dir)
    except Exception as e:
        print(str(e))
        raise RuntimeError
    
    try:
        cfg = get_cfg_from_mlflow(run_id)        
    except Exception as e:
        print(f'{str(e)}')
        raise RuntimeError
    
    try:
        print(f'Tag: {cfg["mlflow"]["tag"]}')
    except Exception as e:
        print(f'{str(e)}')        
                
    device = torch.device('cpu')
    
    # handle cfg compatibility
    cfg = handle_cfg_compatibility(cfg)
    with open_dict(cfg):
        cfg['project_path'] = '.'
        # cfg['training_task'] = 'verification'
    
    train_or_test = 'TRAIN'
    df = pd.read_csv(os.path.join(cfg['project_path'], 'datasets', cfg['data']['df_info_dir'], f'df_{train_or_test}_{cfg["seed"]}.csv'))

    transform_func_list = [broadcast_to_3_chs_album,
                            center_normalization_func(None),
                            ToTensorV2()]
    composed_tranforms = A.Compose(transform_func_list)

    while True:
        try:
            embryo_id = df['embryo_id'].sample(1).values[0]
            filenames = df[df['embryo_id'] == embryo_id].sample(2)['filename'].values
        except:
            pass
        finally:
            break
    
    
    img_paths = [os.path.join(cfg['project_path'], 'datasets', cfg['data']['rel_data_dir'], filename) for filename in filenames]
    rgb_images = [cv2.imread(img_path) for img_path in img_paths]
    gray_images = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in rgb_images]
    
    input_tensor1, input_tensor2 = [composed_tranforms(image=image)['image'] for image in gray_images]
    rgb_image1, rgb_image2 = rgb_images

    network_pl_class = import_network_pl(cfg)    
    # model before training
    model_bf1 = network_pl_class(**{'cfg': cfg})
    model_bf2 = network_pl_class(**{'cfg': cfg})
    # model after training
    checkpoint_path = os.path.join(target_dir_path, [ckpt for ckpt in os.listdir(target_dir_path) if ckpt.endswith('.ckpt')][-1])
    model_af1 = network_pl_class.load_from_checkpoint(checkpoint_path, **{'cfg': cfg})
    model_af2 = network_pl_class.load_from_checkpoint(checkpoint_path, **{'cfg': cfg})        
    
    target_layer_text = ['conv2d_1a',
    'conv2d_2a',
    'conv2d_2b',
    'maxpool_3a',
    'conv2d_3b',
    'conv2d_4a',
    'conv2d_4b',
    'repeat_1',
    'mixed_6a',
    'repeat_2',
    'mixed_7a',
    'repeat_3']

    target_layer_list_bf1 = [getattr(model_bf1, layer) for layer in target_layer_text]
    target_layer_list_bf2 = [getattr(model_bf2, layer) for layer in target_layer_text]
    target_layer_list_af1 = [getattr(model_af1, layer) for layer in target_layer_text]
    target_layer_list_af2 = [getattr(model_af2, layer) for layer in target_layer_text]

    vis_grad_cam(method,
                 model_bf1, model_bf2, 
                 target_layer_list_bf1, target_layer_list_bf2, 
                 input_tensor1, input_tensor2, 
                 rgb_image1, rgb_image2, 
                 use_cuda=gpu,
                 batch_size=batch_size,
                 aug_smooth=aug_smooth,
                 eigen_smooth=eigen_smooth,                 
                 figure_dir=figure_dir,
                 model_phase='before training')
    
    

def vis_grad_cam(method,
                 model_copy1, model_copy2, 
                 target_layer_list1, target_layer_list2, 
                 input_tensor_anc, input_tensor_paired, 
                 rgb_image_anc, rgb_image_paired, 
                 labels=torch.tensor(1),
                 use_cuda=False,
                 batch_size=32,
                 aug_smooth=False,
                 eigen_smooth=False,
                 layer_name='',
                 figure_dir=None,
                 model_phase='after training'):

    cam_algorithm = METHODS[method]
    with cam_algorithm(model_copy1,
                       model_copy2,
                       target_layer_list1,
                       target_layer_list2,                       
                       use_cuda=use_cuda) as cam:
        
        cam.batch_size = batch_size
        grayscale_cam_anc, grayscale_cam_paired = cam(input_tensor_anc,
                                                    input_tensor_paired,
                                                    targets=None,
                                                    aug_smooth=aug_smooth,
                                                    eigen_smooth=eigen_smooth)
        fig, axes = plt.subplots(2, 3, figsize=(20, 5))
        for i, grayscale_cam, rgb_image in [(0, grayscale_cam_anc, rgb_image_anc), 
                                            (1, grayscale_cam_paired, rgb_image_paired)]:                        
            
            cam_img = grayscale_cam[0]                        
            axes[i][0].imshow(cam_img)
            axes[i][0].set_title('heat map cam')                        
            cam_image = show_cam_on_image(rgb_image, cam_img, use_rgb=True)                
            
            axes[i][1].imshow(cam_img)
            axes[i][1].set_title('add on image')            
            
            axes[i][2].imshow(rgb_image)
            axes[i][2].set_title('original image')
            
            title = f'[{method}][layer:{layer_name}][model {model_phase}]'
            plt.suptitle(title)
            plt.savefig(f'{figure_dir}/{title}.png')

if __name__ == '__main__':
    main()        