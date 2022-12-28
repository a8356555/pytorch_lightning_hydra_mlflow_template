import torch
from typing import List, Tuple, Callable
import numpy as np

import sys
sys.append("./pytorch-grad-cam")
from src.models.losses import ContrastiveLoss
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

class WitnessBaseCam(BaseCAM):
    def __init__(self,
                 model_copy1: torch.nn.Module,
                 model_copy2: torch.nn.Module,
                 target_layer_list_from_copy1: List[torch.nn.Module],
                 target_layer_list_from_copy2: List[torch.nn.Module],
                 use_cuda: bool = False,
                 reshape_transform: Callable = None,
                 compute_input_gradient: bool = False,
                 uses_gradients: bool = True) -> None:
        self.model_copy1 = model_copy1.eval()
        self.model_copy2 = model_copy2.eval()
        self.target_layer_list_from_copy1 = target_layer_list_from_copy1
        self.target_layer_list_from_copy2 = target_layer_list_from_copy2
        self.cuda = use_cuda
        if self.cuda:
            self.model_copy1 = self.model_copy1.cuda()
            self.model_copy2 = self.model_copy2.cuda()
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        
        self.activations_and_grads_anc = ActivationsAndGradients(
            self.model_copy1, target_layer_list_from_copy1, reshape_transform)

        self.activations_and_grads_paired = ActivationsAndGradients(
            self.model_copy2, target_layer_list_from_copy2, reshape_transform)
        self.loss_fn = ContrastiveLoss(1)
        self.label = torch.tensor(1) # 一定要 1 不然不會計算

    def forward(self,
                input_tensor_anc: torch.Tensor,
                input_tensor_paired: torch.Tensor,
                targets: List[torch.nn.Module],
                eigen_smooth: bool = False) -> np.ndarray:

        if self.cuda:
            input_tensor_anc = input_tensor_anc.cuda()
            input_tensor_paired = input_tensor_paired.cuda()

        if self.compute_input_gradient:
            input_tensor_anc = torch.autograd.Variable(input_tensor_anc,
                                                       requires_grad=True)
            input_tensor_paired = torch.autograd.Variable(input_tensor_paired,
                                                       requires_grad=True)



        outputs_anc = self.activations_and_grads_anc(input_tensor_anc)
        outputs_paired = self.activations_and_grads_paired(input_tensor_paired)
        
        # if targets is None:
        #     target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
        #     targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            self.model_copy1.zero_grad()
            self.model_copy2.zero_grad()
            loss = self.loss_fn(outputs_anc, outputs_paired, self.label)
            loss.backward(retain_graph=True)

        cam_pairs = []
        for activations_and_grads, target_layer_list, input_tensor in [(self.activations_and_grads_anc, self.target_layer_list_from_copy1, input_tensor_anc),
                                                                       (self.activations_and_grads_paired, self.target_layer_list_from_copy2, input_tensor_paired)]:
            # 與原 module 相容
            self.activations_and_grads = activations_and_grads
            self.target_layers = target_layer_list

            cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                    targets,
                                                    eigen_smooth)
            cam_pairs.append(cam_per_layer)
        return [self.aggregate_multi_layers(cam_per_layer) for cam_per_layer in cam_pairs]


class WitnessGradCAM(GradCAM, WitnessBaseCam):
    pass

class WitnessScoreCAM(ScoreCAM, WitnessBaseCam):
    pass

class WitnessAblationCAM(AblationCAM, WitnessBaseCam):
    pass

class WitnessGradCAMPlusPlus(GradCAMPlusPlus, WitnessBaseCam):
    pass

class WitnessXGradCAM(XGradCAM, WitnessBaseCam):
    pass

class WitnessEigenCAM(EigenCAM, WitnessBaseCam):
    pass

class WitnessEigenGradCAM(EigenGradCAM, WitnessBaseCam):
    pass

class WitnessLayerCAM(LayerCAM, WitnessBaseCam):
    pass

class WitnessFullGrad(FullGrad, WitnessBaseCam):
    pass