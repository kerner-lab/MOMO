
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, vit_l_16, vit_l_32


class ViTB16Encoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(ViTB16Encoder, self).__init__()
        self.encoder = vit_b_16(pretrained=if_pretrained)

    def forward(self, x):
        return self.encoder(x)

class ViTB32Encoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(ViTB32Encoder, self).__init__()
        self.encoder = vit_b_32(pretrained=if_pretrained)

    def forward(self, x):
        return self.encoder(x)

class ViTL16Encoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(ViTL16Encoder, self).__init__()
        self.encoder = vit_l_16(pretrained=if_pretrained)

    def forward(self, x):
        return self.encoder(x)

class ViTL32Encoder(nn.Module):

    def __init__(self, if_pretrained=False):
        super(ViTL32Encoder, self).__init__()
        self.encoder = vit_l_32(pretrained=if_pretrained)

    def forward(self, x):
        return self.encoder(x)


class ViTDecoder(nn.Module):

    def __init__(self, latent_dim=1000, hidden_dim=256, output_channels=3):
        super(ViTDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 14 * 14 * hidden_dim)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 8, output_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 14, 14)
        x = self.decoder(x)
        return x


class ViTAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(ViTAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction
