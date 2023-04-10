import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torchvision.models import resnet18
from module.resnet import resnet20

class ResNet_VAE(nn.Module):
    def __init__(self, num_classes = 10, bottleneck=512):
        super(ResNet_VAE, self).__init__()
        self.encoder = ResNet_Encoder(bottleneck)
        self.decoder = ResNet18Dec(bottleneck=bottleneck)
        self.classifier = ResNet_Classifier(num_classes = num_classes, bottleneck=bottleneck)    

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterization(mean, logvar)
        logits = self.predict(z)

        return z, logits, mean, logvar


    def reconstruct(self, z):
        x_recon = self.decoder(z)

        return x_recon


    def predict(self, z): 
        logits = self.classifier(z)

        return logits

    
    def reparameterization(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        z = torch.distributions.Normal(mu, std+1e-8).rsample()

        return z


class ResNet_Classifier(nn.Module): 
    def __init__(self, num_classes = 10, bottleneck = 64):
        super(ResNet_Classifier, self).__init__()
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(bottleneck,num_classes)
            )


    def forward(self, z):
        logits = self.classifier(z)

        return logits


class ResNet_Encoder(nn.Module):   
    def __init__(self, bottleneck=512):
        super(ResNet_Encoder, self).__init__()
        resnet = resnet18() # Make sure to put bottleneck = 512
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4,
                               stride=2, padding=1, bias=False)
        self.encoder = torch.nn.Sequential(self.conv1,*(list(resnet.children())[1:3]),*(list(resnet.children())[4:-2]))
        ##############################################
        self.conv_mean = nn.Conv2d(bottleneck, bottleneck, kernel_size=2, stride=1, padding=0)
        self.conv_logvar = nn.Conv2d(bottleneck, bottleneck, kernel_size=2, stride=1, padding=0)
        self.fc_mean = nn.Linear(bottleneck,bottleneck)
        self.fc_logvar = nn.Linear(bottleneck,bottleneck)

    def forward(self, x):
        x = self.encoder(x) 
        mean = self.conv_mean(x)
        logvar = self.conv_logvar(x)
        mean = mean.squeeze()
        logvar = logvar.squeeze()
        logvar = logvar.clamp(max=5) # Numerical stability. Equals max(std)==12.1825
        return mean, logvar



class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out



class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], bottleneck=512, nc=3):
        super().__init__()

        self.in_planes = 2*bottleneck

        self.linear = nn.Linear(2*bottleneck, 2*bottleneck)

        self.layer4 = self._make_layer(BasicBlockDec, int(bottleneck), num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, int(bottleneck/2), num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, int(bottleneck/4), num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, int(bottleneck/8), num_Blocks[0], stride=2)
        self.conv1 = ResizeConv2d(int(bottleneck/8), nc, kernel_size=3, scale_factor=2)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        #z = F.relu(self.linear(z))
        z = z.view(z.size(0), z.size(1), 1, 1)
        #x = F.interpolate(x, scale_factor=2)
        x = self.layer4(z)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        #x = x.view(x.size(0), 3, 32, 32)
        return x
