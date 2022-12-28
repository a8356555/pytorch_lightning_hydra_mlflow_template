import os
import numpy as np 
import pandas as pd

from src.models.datamodule.dataset_base import DatasetBase


class Dataset_custom(DatasetBase):
    def __init__(self, 
    cfg, 
    transform=None, 
    train=False
    ):
        super().__init__(cfg, transform, train, is_img_gray=True)        
        pass
