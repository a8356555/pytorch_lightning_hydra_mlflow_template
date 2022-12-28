from tokenize import Triple
import pytorch_lightning as pl
import time
import numpy as np
import torch
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.nn as nn
import os
import sys
network_pl = sys.modules[__name__]

from src.models import losses
from pytorch_metric_learning import losses as pml_losses
from src.models.margin_schedulers import *
from src.utils.dynamic_import import import_backbone, import_head
from src.utils.evaluator import TripletEvaluator
from src.utils.utils import createLogHandler
from src.models.backbone.mae_vit import interpolate_pos_embed

pylogger = createLogHandler(__name__, 'training_log.log')

def get_backbone_by_cfg(cfg):
    backbone_class = import_backbone(cfg['backbone'])
    backbone_kwargs = cfg['backbone']
    backbone_kwargs = {k: v for k, v in backbone_kwargs.items() if k not in ['file_name', 'class_name', 'use_pretrained', 'pretrained_file', 'backbone_trainable']}
    backbone = backbone_class(**backbone_kwargs)
    
    if cfg['backbone'].get('use_pretrained') and cfg['backbone'].get('pretrained_file'):        
        print('pretrained backbone loaded')
        state_dict = torch.load(os.path.join(cfg['project_path'], 'model_weights', cfg['backbone']['pretrained_file']), map_location='cpu' if cfg['gpus'] == 0 else None)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']            
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        backbone.load_state_dict(state_dict, strict=False)
    print(backbone)
    return backbone


def get_head_by_cfg(cfg):
    if cfg.get('head'):
        head_class = import_head(cfg['head'])
        head_kwargs = {k:v for k, v in cfg['head'].items() if 'name' not in k}        
        in_features=cfg['backbone']['embedding_size']
        out_features=cfg['data']['num_classes']
        return head_class(in_features, out_features, **head_kwargs)
    else:
        return None
        
def get_loss_by_cfg(cfg, for_verification=False):
    return getattr(losses, cfg['loss']['name'])(cfg)
    
        
def get_optimizer_by_cfg(cfg, param_group):
    """[summary]

    Args:
        cfg ([type]): [description]
        param_group ([dict, list]): [description]

    Returns:
        [type]: [description]
    """
    optimizer_class = getattr(network_pl, cfg['optimizer']['name'])
    optimizer_kwargs = {k:v for k, v in cfg['optimizer'].items() if k not in ['name', 'layer_to_update']}    
    return optimizer_class(param_group, **optimizer_kwargs)

def get_lr_scheduler_by_cfg(cfg, optimizer):    
    lr_sch_class = getattr(network_pl, cfg['lr_scheduler']['name'])
    lr_sch_kwargs = {k:v for k, v in cfg['lr_scheduler'].items() if k not in ['name', ]}
    return lr_sch_class(optimizer, **lr_sch_kwargs)



class ModelBase_pl(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg
        self.backbone = get_backbone_by_cfg(cfg)        
        self.lr_scheduler_enabled = cfg.get('lr_scheduler')
        self.evaluator = None

    def forward(self, x):
        return self.backbone(x)

    def get_embedding(self, x):
        return self.backbone(x)

    def on_train_start(self):
        pylogger.info(self.optimizers())

    def on_epoch_start(self):
        self.train_size = len(self.trainer.datamodule.train_dataloader().dataset)
        self.val_size = len(self.trainer.datamodule.val_dataloader().dataset)

        self.train_total_batches = len(self.trainer.datamodule.train_dataloader())
        self.val_total_batches = len(self.trainer.datamodule.val_dataloader())

    # training
    def on_train_epoch_start(self):
        self.start = time.time()
        self.losses = torch.zeros(self.train_total_batches)        
        self.train_total_loss = 0

    def training_step(self, batch, batch_idx):
        pass

    def training_epoch_end(self, dict_per_epoch):
        pass

    # validation
    def on_validation_epoch_start(self):
        self.start = time.time()
        self.val_total_loss = 0

    def validation_step(self, val_batch, batch_idx):        
        pass

    def validation_epoch_end(self, dict_per_epoch):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):        
        param_group = self._get_param_group()
        optimizer = get_optimizer_by_cfg(self.cfg, param_group)
        opt_dict = {'optimizer': optimizer}

        pylogger.info('*' * 10 + 'Optimizer Configured!' + '*' * 10)
        pylogger.info(optimizer)
        
        if self.lr_scheduler_enabled:
            lr_scheduler = get_lr_scheduler_by_cfg(self.cfg, optimizer)
            opt_dict['lr_scheduler'] = {
                'scheduler': lr_scheduler,
                'monitor': self.cfg['checkpoint']['monitor'], 
                'interval': 'epoch'
            }
            pylogger.info('*' * 10 + 'Lr scheduler Configured!' + '*' * 10)
            pylogger.info(lr_scheduler)        
        
        return opt_dict
        
    def _get_param_group(self):                
        backbone_parameters = getattr(self, ''.join(('_get_params_', self.cfg['optimizer']['layer_to_update'])))()
        return [
            {'params': backbone_parameters},            
        ]
    def _get_params_all(self):
        return self.parameters()
        
    def _get_params_last(self):
        last_linear_ids = list(map(id, self.backbone.last_linear.parameters()))
        last_bn_ids = list(map(id, self.backbone.last_bn.parameters()))
        return filter(lambda p: id(p) in last_linear_ids + last_bn_ids, self.parameters())
    
    def _get_params_last_linear(self):
        return self.last_linear.parameters()

class Network_pl(ModelBase_pl):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        self.head = get_head_by_cfg(cfg)
        self.loss_fn =  get_loss_by_cfg(cfg)

    def _process_batch(self, batch):
        x, y, extra_info = batch
        # return x.half(), y
        return x, y, np.array(extra_info)

    def training_step(self, batch, batch_idx):
        out = self.backbone(x)        
        # if self.head:
        #     embeddings = self.head(embeddings, y)
        
        pass
    
    def training_epoch_end(self, outputs):
        pass
    
    def validation_step(self, val_batch, batch_idx):
        pass
    
    def validation_epoch_end(self, val_outputs):
        pass