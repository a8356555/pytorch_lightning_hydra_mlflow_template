from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import sys
datamodule_pl = sys.modules[__name__]

from src.utils.dynamic_import import import_dataset
from src.visualization.visualizer import save_dataset_example_as_png
from src.utils.utils import createLogHandler

pylogger = createLogHandler(__name__, 'training_log.log')

def center_normalization_func(cfg):
    # img = (img - mean * max_pixel_value) / ( std * max_pixel_value)
    return A.Normalize(mean=(127.5/128.0, 127.5/128.0, 127.5/128.0), std=(1.0, 1.0, 1.0), max_pixel_value=128.0, p=1.0)

def custom_normalization_func(cfg):
    return A.Normalize(mean=(cfg['aug_mean']), std=(cfg['aug_std']), max_pixel_value=255.0, p=1.0)

def normalize_normalization_func(cfg):
    return A.Normalize(mean=[0,0,0], std=[1,1,1], max_pixel_value=255.0, p=1.0)


def imagenet_normalization_func(cfg):
    return A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0)


def center_crop_func(cfg):
    return A.CenterCrop(*cfg['aug_crop_h_w'], p=cfg['aug_crop_prob'])

def random_crop_func(cfg):
    return A.RandomCrop(*cfg['aug_crop_h_w'], p=cfg['aug_crop_prob'])

def get_resize_func(cfg):
    if cfg['aug_resize_method'] == 'normal':
        return A.Resize(*cfg['aug_resize_h_w'], p=cfg['aug_resize_prob'])
    elif cfg['aug_resize_method'] == 'smallest':
        # choose max size from list cfg['aug_resize_h_w]
        return A.SmallestMaxSize(cfg['aug_resize_h_w'], p=cfg['aug_resize_prob'])
    elif cfg['aug_resize_method'] == 'longest':
        return A.LongestMaxSize(cfg['aug_resize_h_w'], p=cfg['aug_resize_prob'])

def broadcast_to_3_chs_album(image=None, force_apply=True):         
    shape = image.shape 
    if len(shape) == 2:
        image = np.expand_dims(image, -1)        
        return {'image': np.broadcast_to(image, (*shape, 3))}    
    else:
        raise ValueError(f'Image should be gray scale, the image shape and channels is however: {image.shape}, {image.shape[2]}')

def get_composed_transforms(cfg, is_train=True):
    gamma_limit = tuple(int(val*100) for val in cfg['aug_gamma_range'])
    shift, scale, rotate = cfg['aug_ssr_value']    
    
    broadcast_gray_scale = broadcast_to_3_chs_album if cfg['aug_broadcast_gray_scale'] else None
    resize_func = get_resize_func(cfg)
    crop_func = getattr(datamodule_pl, ''.join((cfg['aug_crop_method'], '_crop_func')))(cfg)    
    hflip_func = A.HorizontalFlip(p=cfg['aug_h_v_flip_prob'][0])
    vflip_func = A.VerticalFlip(p=cfg['aug_h_v_flip_prob'][1])
    random_gamma_func = A.RandomGamma(gamma_limit=gamma_limit, p=cfg['aug_gamma_prob'])    
    shift_scale_rotate = A.ShiftScaleRotate(shift_limit=shift, scale_limit=scale, rotate_limit=rotate, p=cfg['aug_ssr_prob'])
    normalize = getattr(datamodule_pl, ''.join((cfg['aug_norm_method'], '_normalization_func')))(cfg)

    if is_train:
        transform_func_list = [broadcast_gray_scale,
                            resize_func,
                            crop_func,
                            hflip_func,
                            vflip_func,
                            random_gamma_func,
                            shift_scale_rotate,
                            normalize,
                            ToTensorV2()]
    else:
        transform_func_list = [broadcast_gray_scale,
                            resize_func,
                            crop_func,
                            normalize,
                            ToTensorV2()]
    pylogger.info(f'Composed transforms: {transform_func_list}')
    composed_tranforms = A.Compose([func for func in transform_func_list if func is not None])
    return composed_tranforms                

def get_dataloader_for_specific_task(cfg, is_train=True, custom_dataset_cls_prefix=None):    
    train_or_test = 'train' if is_train else 'test'    
    pylogger.info('*' * 10 + f'Start loading {train_or_test} data' + 10 * '*')
    
    composed_transforms = get_composed_transforms(cfg, is_train=is_train)
    dataset_class = import_dataset(cfg, is_train=is_train, custom_dataset_cls_prefix=custom_dataset_cls_prefix)
    dataset = dataset_class(
        cfg, 
        transform=composed_transforms, 
        train=is_train
    )

    dataloader_kwargs = {'num_workers': cfg['num_workers'], 'pin_memory': cfg['pin_memory']}
    shuffle = cfg['is_train_shuffled'] if is_train else False
    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=shuffle, **dataloader_kwargs)

    pylogger.info(dataloader.dataset.__class__)
    pylogger.info(dataloader.dataset.get_info())    
    is_class_showed = False if is_train else True
    save_dataset_example_as_png(cfg, dataloader.dataset, train_or_test=train_or_test, is_class_showed=is_class_showed)
    return dataloader

class DataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.train_loader = get_dataloader_for_specific_task(self.cfg, is_train=True)
        self.val_loader = get_dataloader_for_specific_task(self.cfg, is_train=False)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader