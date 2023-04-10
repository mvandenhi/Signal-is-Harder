''' Modified from https://github.com/alinlab/LfF/blob/master/module/util.py '''

import torch.nn as nn
from module.resnet import resnet20
from module.mlp import MLP
from module.mlp_vae import MLP_VAE
from module.resnet_vae import ResNet_VAE
from torchvision.models import resnet18, resnet50

def get_model(config):
    model_tag = config["model"]["tag"]
    dataset = config["data"]["dataset"]
    if dataset in {"colored_mnist", "cifar10_type0", "cifar10_type1"}:
        num_classes = 10
    elif dataset in {"camelyon17_type0", "camelyon17_type1", "camelyon17_type2"}:
        num_classes = 2
    else: raise NotImplementedError("Dataset is not integrated.")
    
    if model_tag == "ResNet20":
        return resnet20(num_classes)
    elif model_tag == "ResNet18":
        model = resnet18(pretrained=True)
        print("Pretrained&frozen ResNet18")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model
    elif model_tag == "ResNet50":
        model = resnet50(pretrained=True)
        print("Pretrained&frozen ResNet50")
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "MLP_VAE":
        return MLP_VAE(num_classes=num_classes,bottleneck=config["model"]["bottleneck_MLP"])
    elif model_tag == "ResNet_VAE":
        return ResNet_VAE(num_classes=num_classes,bottleneck = config["model"]["bottleneck_ResNet"])
    else:
        raise NotImplementedError("Model not implemented.")



# def get_disentangler(config):
#     model_tag = config["model"]["tag"]    
#     if model_tag == "MLP":
#         return FFVAE_Disentangler()
#     elif model_tag == "MLP_VAE":
#         return FFVAE_Disentangler()
#     else:
#         raise NotImplementedError("Model not implemented.")