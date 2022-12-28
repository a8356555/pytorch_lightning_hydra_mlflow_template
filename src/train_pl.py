from mlflow.protos.service_pb2 import Run
import torch
import numpy as np
import os
import shutil
from datetime import datetime
import hydra
import mlflow
from omegaconf import DictConfig    
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import traceback
from pathlib import Path

from src.utils.dynamic_import import import_network_pl
from src.models.datamodule.datamodule_pl import DataModule
from src.utils.utils import createLogHandler, handle_and_check_cfg, handle_mlflow_log_param

lr_monitor = LearningRateMonitor(logging_interval='step')
pylogger = createLogHandler(__name__, 'training_log.log')

def get_model(cfg, ckpt_description, cur_run_id):
    network_pl_class = import_network_pl(cfg)
    if cfg['resume_from_checkpoint']['enable']:
        previous_run_id = cfg['resume_from_checkpoint']['previous_run_id']
        old_checkpoint_dir = list((Path(cfg['project_path']) / 'model_weights').glob(f'*{previous_run_id}'))[-1]        
        old_checkpoint_path = os.path.join(old_checkpoint_dir, os.listdir(old_checkpoint_dir)[-1])
        model = network_pl_class.load_from_checkpoint(old_checkpoint_path, **{'cfg': cfg}, strict=False)
        checkpoint_dir = str(old_checkpoint_dir).replace('[Keep]', '') + '_' + str(cur_run_id)
        pylogger.info(f'{old_checkpoint_path} , old model loaded')
    else: 
        checkpoint_dir = os.path.join(cfg['project_path'], 'model_weights', ckpt_description)
        old_checkpoint_path = None
        model = network_pl_class(cfg)
    return model, network_pl_class, checkpoint_dir, old_checkpoint_path


def get_trainer(cfg, ckpt_description, checkpoint_dir, old_checkpoint_path):
    logger_dir = os.path.join(cfg['project_path'], 'logs', ckpt_description)
    logger = TensorBoardLogger(save_dir=logger_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['checkpoint']['monitor'],
        dirpath=checkpoint_dir,
        filename='{epoch}{'+cfg['checkpoint']['monitor']+':.2f}',
        save_top_k = cfg['checkpoint']['save_top_k_models'],
        every_n_epochs=cfg['checkpoint']['save_every_n_epoch'],
        mode=cfg['checkpoint']['mode'],
        save_weights_only=True,            
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        logger=logger,        
        max_epochs=cfg['n_epochs'], 
        gpus=cfg['gpus'] if cfg['is_cuda_available'] else None, 
        amp_backend=cfg['mixed_precision']['backend'],
        precision=cfg['mixed_precision']['precision'] if cfg['mixed_precision']['enable'] else 32,
        amp_level=cfg['mixed_precision']['amp_level'] if cfg['mixed_precision']['enable'] else None,
        log_every_n_steps=cfg['logger']['log_every_n_steps'],
        callbacks=[checkpoint_callback, lr_monitor],
        accumulate_grad_batches=max(int(cfg['global_batch_size']/cfg['batch_size']), 1) if 'global_batch_size' in cfg.keys() else 1
        # resume_from_checkpoint=old_checkpoint_path if cfg['resume_from_checkpoint']['is_resumed'] else None # due to save weight only, this will fail
    )
    return trainer
    
@hydra.main(config_path='configs', config_name='config')
def main(cfg: DictConfig) -> None:       
    seed = cfg['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # 如果卷积网络结构不是动态变化的，网络的输入 (batch size，图像的大小，输入的通道) 是固定的 就True
    torch.backends.cudnn.benchmark = True    
    torch.backends.cudnn.enabled = True
    # torch.backends.cudnn.deterministic = True # 這會讓網路變慢，但每次返回的捲雞結果一樣
    
    pylogger.info(f'')
    pylogger.info(f'Start! {seed} Seeded')
    
    try: # handle cfg
        cfg = handle_and_check_cfg(cfg)
    except Exception as e:
        pylogger.error('Cfg Handling and Checking Failed : '+ str(e))
        raise RuntimeError

    device = torch.device('cuda:0' if cfg['is_cuda_available'] else 'cpu')    
    pylogger.info('Running on device: {}'.format(device))

    mlflow.set_tracking_uri(cfg['mlflow']['url'])
    mlflow.set_experiment(cfg['mlflow']['experiment_name'])
        
    try: # load datamodule
        datamodule = DataModule(cfg)
    except Exception as e:
        pylogger.error('Datamodule Loading Failed : '+ str(e))
        print(traceback.format_exc())
        raise RuntimeError
    
    pylogger.info('*' * 10 + 'datamodule Loaded' + '*' * 10)

    mlflow.pytorch.autolog()
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        pylogger.info(f'run_id: {run_id}')
        if cfg['mlflow']['tag'] is not None:
            print('exp: ', cfg['mlflow']['tag'])
            mlflow.set_tag('exp', cfg['mlflow']['tag'])
        mlflow.set_tag('run_id', run_id)

        ckpt_description = ('Model_{}_{}'.format(cfg['backbone']['class_name'], datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        ckpt_description = ckpt_description + '_' + run_id

        try: # load model
            model, network_pl_class, checkpoint_dir, old_checkpoint_path = get_model(cfg, ckpt_description, run_id)
        except Exception as e:
            pylogger.error('Model Loading Failed : '+ str(e))
            raise RuntimeError

        pylogger.info('*' * 10 + 'Model Loaded' + '*' * 10)
        # print(model)
        
        try: # load trainer
            trainer = get_trainer(cfg, ckpt_description, checkpoint_dir, old_checkpoint_path)
        except Exception as e:
            pylogger.error('Trainer Loading Failed : '+ str(e))
            raise RuntimeError

        try: # handle mlflow log param
            handle_mlflow_log_param(cfg)
        except Exception as e:
            pylogger.error('mlflow log param Failed : '+ str(e))

        pylogger.info('*' * 10 + 'Start Training' + '*' * 10)
        
        try: # load trainer
            trainer.fit(model, datamodule)
        except Exception as e: # work on python 2.x
            pylogger.error('Traing Failed : '+ str(e))
            raise RuntimeError

        pylogger.info('*' * 10 + 'End Training' + '*' * 10)

    # load model
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, os.listdir(checkpoint_dir)[-1])
    
    try: # load trained model
        model = network_pl_class.load_from_checkpoint(checkpoint_path, **{'cfg': cfg}) 
    except Exception as e:
        pylogger.error('Trained Model Loading Failed : '+ str(e))
    finally:
        pylogger.info('*' * 10 + 'Trained Model Loaded' + '*' * 10)

    pylogger.info('Ended!')



if __name__ == '__main__':
    main()
