
import torch.nn as nn
from torchvision.models import resnet34

class ResNetEncoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(ResNetEncoder, self).__init__()
        resnet = resnet34(pretrained=if_pretrained)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, x):
        return self.encoder(x)

class ResNetDecoder(nn.Module):

    def __init__(self, input_channels=512):
        super(ResNetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


class ResNetAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(ResNetAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
