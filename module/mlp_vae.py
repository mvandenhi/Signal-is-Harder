import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_VAE(nn.Module):
    def __init__(self, num_classes = 10, bottleneck = 16):
        super(MLP_VAE, self).__init__()
        self.encoder = MLP_Encoder(bottleneck = bottleneck)
        self.decoder = MLP_Decoder(bottleneck = bottleneck)
        self.classifier = MLP_Classifier(num_classes = num_classes, bottleneck = bottleneck)    

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


class MLP_Encoder(nn.Module):   
    def __init__(self, bottleneck = 16):
        super(MLP_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3*28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.fc_mean = nn.Linear(100, bottleneck)
        self.fc_logvar = nn.Linear(100, bottleneck)


    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        return mean, logvar



class MLP_Decoder(nn.Module):
    def __init__(self, bottleneck = 16):
        super(MLP_Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck*2, 512), # Combined representations of signal and bias encoder
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3*28*28)
        )

    def forward(self, z):
        z = self.decoder(z)
        x_hat = torch.sigmoid(z)
        x_hat = x_hat.view(x_hat.size(0),3,28,28)
        
        return x_hat



class MLP_Classifier(nn.Module): 
    def __init__(self, num_classes = 10, bottleneck = 16):
        super(MLP_Classifier, self).__init__()
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(bottleneck,num_classes),
            )

        self.num_classes = num_classes #Necessary for DeepFool2

    def forward(self, z):
        logits = self.classifier(z)

        return logits
        