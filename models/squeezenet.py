
import torch.nn as nn
from torchvision.models import squeezenet1_1

class SqueezeNetEncoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(SqueezeNetEncoder, self).__init__()
        squeezenet = squeezenet1_1(pretrained=if_pretrained)
        self.encoder = nn.Sequential(*list(squeezenet.children())[:-1])

    def forward(self, x):
        return self.encoder(x)

class SqueezeNetDecoder(nn.Module):

    def __init__(self, input_channels=512):
        super(SqueezeNetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 256, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=8, stride=1, padding=3),
        )

    def forward(self, x):
        return self.decoder(x)


class SqueezeNetAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(SqueezeNetAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
