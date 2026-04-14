from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

#############################################################################
# VGG16
#############################################################################
def make_vgg16():
    weights = VGG16_Weights.DEFAULT
    model = vgg16(weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    return model, weights.transforms()

#############################################################################
# Efficientnet B0
#############################################################################
def make_effnet_B0():
    weights = EfficientNet_B0_Weights.DEFAULT
    transforms = weights.transforms()
    base_model = efficientnet_b0(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
        
    return base_model, transforms


#############################################################################
# Resnet18
#############################################################################
def make_resnet18():
    weights = ResNet18_Weights.DEFAULT
    transforms = weights.transforms()
    base_model = resnet18(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
        
    return base_model, transforms


#############################################################################
# MobileNet3 small
#############################################################################
def make_mobilenet3_small():
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transforms = weights.transforms()
    base_model = mobilenet_v3_small(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
    
    return base_model, transforms