
import torch.nn as nn
from torchvision.models import efficientnet_v2_m

class EfficientNetEncoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(EfficientNetEncoder, self).__init__()
        efficientnet = efficientnet_v2_m(pretrained=if_pretrained)
        self.encoder = nn.Sequential(*list(efficientnet.children())[:-2])

    def forward(self, x):
        return self.encoder(x)

class EfficientNetDecoder(nn.Module):

    def __init__(self, input_channels=1280):
        super(EfficientNetDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_channels, 512, kernel_size=4, stride=2, padding=1),  # 1280 → 512
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),   # 512 → 256
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),   # 256 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),    # 128 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        return self.decoder(x)


class EfficientNetAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EfficientNetAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
