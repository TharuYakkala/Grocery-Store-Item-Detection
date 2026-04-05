from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

#############################################################################
# VGG16
#############################################################################
def make_vgg16(device):
    weights = VGG16_Weights.IMAGENET1K_V1
    model = vgg16(weights=weights)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=32, bias=True),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features=32, out_features=5)
    )

    model = model.to(device)
    return model, weights.transforms()

#############################################################################
# Efficientnet B0
#############################################################################
def make_effnet_B0(device):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    transforms = weights.transforms()
    base_model = efficientnet_b0(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
        
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.75),
        nn.Linear(in_features=1280, out_features=5, bias=True)
    )
    # for layer in base_model.features[-1:]:
    #     for param in layer.parameters():
    #         param.requires_grad = True
    base_model = base_model.to(device)
    return base_model, transforms


#############################################################################
# Resnet18
#############################################################################
def make_resnet18(device):
    weights = ResNet18_Weights.DEFAULT
    transforms = weights.transforms()
    base_model = resnet18(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
        
    base_model.fc = nn.Sequential(
        nn.Dropout(0.75),
        nn.Linear(in_features=512, out_features=5, bias=True)
    )
    # for layer in base_model.features[-1:]:
    #     for param in layer.parameters():
    #         param.requires_grad = True
    base_model = base_model.to(device)
    return base_model, transforms


#############################################################################
# MobileNet3 small
#############################################################################
def make_mobilenet3_small(device):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transforms = weights.transforms()
    base_model = mobilenet_v3_small(weights=weights)
    for param in base_model.parameters():
        param.requires_grad = False
        
    base_model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features=576, out_features=5, bias=True)
    )
    # for layer in base_model.features[-1:]:
    #     for param in layer.parameters():
    #         param.requires_grad = True
    base_model = base_model.to(device)
    return base_model, transforms