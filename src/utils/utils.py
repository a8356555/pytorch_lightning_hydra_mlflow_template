import json
import os
from omegaconf import OmegaConf, open_dict, dictconfig
import hydra
import torch
import mlflow
import logging
import ast

BASIC_NEEDED_CONFIG = {
    'data': {},    
    'backbone': {},
    'loss': {},
    'batch_size': 128,
    'num_workers': 4,
    'pin_memory': True,
    'is_train_shuffled': True,
    'mp_pre_read': False,
    'aug_broadcast_gray_scale': True,
    'aug_resize_method': 'normal',
    'aug_resize_prob': 0.,
    'aug_resize_h_w': [120, 160],
    'aug_crop_method': 'center',
    'aug_crop_prob': 0.,
    'aug_crop_h_w': [120, 120],
    'aug_h_v_flip_prob': [0., 0.],
    'aug_norm_method': 'center',
    'aug_mean': None,
    'aug_std': None,
    'aug_ssr_value': [0.1, 0.05, 45],
    'aug_ssr_prob': 0.,
    'aug_gamma_range': [0.9, 1.1],
    'aug_gamma_prob': 0.,
    }

def createLogHandler(job_name,log_file):
    logger = logging.getLogger(job_name)
    ## create a file handler ##
    handler = logging.FileHandler(log_file)
    ## create a logging format ##
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    handler.setFormatter(logFormatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

def save_cfg(cfg, save_dir):
    target_path = os.path.join(save_dir, 'config.json')
    output_dict = OmegaConf.to_container(cfg)    
    with open(target_path, 'w') as out_file:
        json.dump(output_dict, out_file, ensure_ascii=False, indent=4)
    print('*' * 10 + 'Config Saved' + '*' * 10)

def load_cfg(target_dir_path):
    PATH = os.path.join(target_dir_path, 'config.json')
    with open(PATH, 'r') as in_file:
        data = json.load(in_file)
    return data

class CfgChecker:
    def __init__(self, cfg):
        self.cfg = cfg


    def _check_cfg_optim_layer_to_update(self):
        layer_to_update_list = ['all', 'last', 'last_linear', 'custom_last_linear']
        if self.cfg['optimizer']["layer_to_update"] not in layer_to_update_list:
            raise RuntimeError(f'layer_to_update should be in following: {layer_to_update_list}')

    def check(self):
        for attr_name in dir(self):
            if attr_name.startswith('_check_'):
                getattr(self, attr_name)()
        return self.cfg

class CfgCompatibilityHandler:
    def __init__(self, cfg):        
        self.cfg = cfg

    def _eval_input_value(self, input_value):
        try:
            input_value = ast.literal_eval(input_value)
        except Exception as e:
            # type str will cause error
            pass
        return input_value
            
    def _handle_lost_nested_config(self, key, sub_dict):
        for sub_key in sub_dict.keys():
            if key not in self.cfg.keys() or sub_key not in self.cfg[key].keys():
                input_value = input(f'Please enter [cfg.{key}.{sub_key}] value in python data format,\nif nothing input then default value "{BASIC_NEEDED_CONFIG[key][sub_key]}" will be used: \n')
                input_value = BASIC_NEEDED_CONFIG[key][sub_key] if not input_value else self._eval_input_value(input_value)
                self.cfg[key][sub_key] = input_value

    def _handle_lost_config(self, key):
        if key not in self.cfg.keys() or key is None:
            input_value = input(f'Please enter [cfg.{key}] value in python data format,\nif nothing input then default value "{BASIC_NEEDED_CONFIG[key]}" will be used (if you change config file then check whether the BASIC_NEEDED_CONFIG in src/utils/utils.py is changed too): \n')
            input_value = BASIC_NEEDED_CONFIG[key] if not input_value else self._eval_input_value(input_value)
            self.cfg[key] = input_value

    def handle_all_lost_config(self):
        with open_dict(self.cfg):
            for key, val in BASIC_NEEDED_CONFIG.items():
                try:
                    if isinstance(val, dict):
                        self._handle_lost_nested_config(key, val)
                    else:
                        self._handle_lost_config(key)                    
                except:
                    pass
        return self.cfg

    def handle_None_value(self):
        with open_dict(self.cfg):
            self.cfg = self._recursive_handle_none_value(self.cfg)
        return self.cfg
    
    def _recursive_handle_none_value(self, temp_dict):
        for key, val in temp_dict.items():
            if isinstance(val, dictconfig.DictConfig):
                temp_dict[key] = self._recursive_handle_none_value(val)        
            else:
                if val == 'None':
                    temp_dict[key] = None
        return temp_dict


def handle_cfg_compatibility(cfg):
    cfg_compatibility_handler = CfgCompatibilityHandler(cfg)
    cfg = cfg_compatibility_handler.handle_all_lost_config()
    cfg = cfg_compatibility_handler.handle_None_value()
    return cfg

def check_cfg(cfg):
    cfg_checker = CfgChecker(cfg)
    cfg = cfg_checker.check()
    return cfg

def handle_and_check_cfg(cfg):
    with open_dict(cfg):
        cfg['project_path'] = hydra.utils.get_original_cwd()    
        cfg["is_cuda_available"] = torch.cuda.is_available()
    cfg = handle_cfg_compatibility(cfg)
    cfg = check_cfg(cfg)
    return cfg

def _transfer_param_dict_to_confdict(param_dict):
    import re
    import ast
    for key, value in param_dict.items():    
        try:
            if value in ['True', 'False'] or not re.search('[a-zA-Z]', value):
                param_dict[key] = ast.literal_eval(value)
        except Exception as e:        
            print(f'{key}: {value}, message: {str(e)}')
        if re.search('{', value):
            param_dict[key] = OmegaConf.create(value) 
    cfg = OmegaConf.create(param_dict)
    return cfg
    

def get_cfg_from_mlflow(run_id):
    from mlflow.tracking import MlflowClient   

    MLFLOW_URL = 'http://0.0.0.0:5000/'
 
    client = MlflowClient(tracking_uri=MLFLOW_URL)
    target_run = client.get_run(run_id)
    param_dict = target_run.data.params
    cfg = _transfer_param_dict_to_confdict(param_dict)

    return cfg


def handle_mlflow_log_param(cfg):
    # make mlflow search easier
    mlflow.log_params({'backbone_name': cfg['backbone']['class_name']})    

    for k, v in cfg.items():
        mlflow.log_params({k: v})

def make_custom_network_cfg(
    pretrained=None, 
    dropout_prob=0.6, 
    device=torch.device('cpu'),
    backbone_file_name='inception_resnet_v1',
    backbone_class_name='InceptionResnetV1',
    embedding_size=512,
    loss_margin=1
    ):
    cfg = {'backbone': {
                'pretrained': pretrained, 
                'dropout_prob': dropout_prob, 
                'device': device,
                'file_name': backbone_file_name,
                'class_name': backbone_class_name,
                'embedding_size': embedding_size
                },
            'loss': {
                'margin': loss_margin
                }
            }
    return cfg