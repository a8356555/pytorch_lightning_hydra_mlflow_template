import torch
from torch.nn import *
import torch.nn.functional as F
from pytorch_metric_learning.miners import *


class CustomLoss(nn.Module):
    def __init__(self, cfg):
        pass