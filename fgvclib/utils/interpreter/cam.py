import typing as t
import numpy as np
import os.path as op
import os
import cv2
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn, Tensor

from .interpreter import Interpreter


class CAM(Interpreter):

    methods = {
        "gradcam": GradCAM,
        "hirescam":HiResCAM,
        "scorecam": ScoreCAM,
        "gradcam++": GradCAMPlusPlus,
        "ablationcam": AblationCAM,
        "xgradcam": XGradCAM,
        "eigencam": EigenCAM,
        "eigengradcam": EigenGradCAM,
        "layercam": LayerCAM,
        "fullgrad": FullGrad,
        "gradcamelementwise": GradCAMElementWise
    }
    
    def __init__(self, model:nn.Module, target_layers:list, use_cuda:bool, method:str="gradcam", aug_smooth:bool=True, eigen_smooth:bool=True, save_dir:str="./cam_images") -> None:
        super(CAM, self).__init__(model)
        self.target_layers = target_layers

        # if isinstance(model, nn.Module):
        #     target_layers=[model.layer4]
        # else:
        target_layers=[model.module.backbone.layer4]
        self.interpreter = GradCAM(model=model, target_layers=target_layers, use_cuda=use_cuda)
        self.aug_smooth = aug_smooth
        self.eigen_smooth = eigen_smooth
        self.cam_algorithm = self.methods[method]
        self.save_dir = save_dir
        if not op.exists(save_dir):
            os.mkdir(save_dir)

    def __call__(self, image_path, image_tensor:Tensor, transforms, targets:t.List[int]=None, save:bool=False) -> np.ndarray:

        rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
        rgb_img = np.float32(rgb_img) / 255
        
        t_h, t_w = image_tensor.shape[-2:]
        i_h, i_w = rgb_img.shape[:2]

        rgb_img = cv2.resize(src=rgb_img, dsize=(t_w, t_h))

        grayscale_cam = self.interpreter(input_tensor=image_tensor, targets=targets, aug_smooth=self.aug_smooth, eigen_smooth=self.eigen_smooth)

        grayscale_cam = grayscale_cam[0, :]
        
        if save:
            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            cv2.imwrite(op.join(self.save_dir, f'{op.basename(image_path)}_cam.jpg'), cam_image)
    
        grayscale_cam = cv2.resize(src=grayscale_cam, dsize=(i_w, i_h))

        return grayscale_cam

    @classmethod
    def available_methods(self):
        return self.methods


def cam(model, cfg) -> CAM:
    
    return CAM(
        model=model, 
        target_layers=cfg.INTERPRETER.TARGET_LAYERS, 
        use_cuda=cfg.USE_CUDA, 
        method=cfg.INTERPRETER.METHOD
    )
    
