from timm.models.layers import DropPath
from einops import rearrange
import numpy as np
import random
import torch
from torch import nn
import os
import math
import torch.nn.functional as F

from timm.models.layers import trunc_normal_

from models_pretrain import *
from models.models_vit import vit_customized
from utils.pos_embed import interpolate_pos_embed

# Works exactly the same as before - just better results!
# model = Segmentation_ViT_UNetFormer(encoder=vit_encoder, encoder_output_dim=768, num_classes=1, decoder_channels=64, window_size=8, dropout=0.1)

# Forward pass unchanged
# output = model(images)  # (B, 1, H, W)
# ============================================================================
# UNetFormer Decoder Components
# ============================================================================


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.ws - 1
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps, 0, 0), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        out = out[:, :, :H, :W]

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
                                nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels//16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels//16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


class UNetFormerDecoder(nn.Module):
    """
    IMPROVED UNetFormer decoder with multi-scale feature processing.
    
    Enhancements:
    - Multi-stage decoder with progressive upsampling (4x -> 2x -> 1x channels)
    - Multiple Global-Local Attention blocks at different scales
    - Multi-scale feature fusion for better boundary detection
    - Prevents object merging by capturing features at multiple resolutions
    """
    def __init__(self,
                 in_channels=768,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=1):
        super(UNetFormerDecoder, self).__init__()

        # Multi-scale feature extraction - start with 4x channels
        self.pre_conv = ConvBN(in_channels, decode_channels * 4, kernel_size=1)
        
        # Stage 1: Finest scale (4x channels)
        self.b4 = Block(dim=decode_channels * 4, num_heads=8, window_size=window_size)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(decode_channels * 4, decode_channels * 2, kernel_size=3)
        )
        
        # Stage 2: Medium scale (2x channels)
        self.b3 = Block(dim=decode_channels * 2, num_heads=4, window_size=window_size)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBNReLU(decode_channels * 2, decode_channels, kernel_size=3)
        )
        
        # Stage 3: Coarsest scale (1x channels)
        self.b2 = Block(dim=decode_channels, num_heads=2, window_size=window_size // 2)
        
        # Multi-scale fusion
        # Concatenate: 4x + 2x + 1x = 7x channels
        self.fusion = nn.Sequential(
            ConvBNReLU(decode_channels * 7, decode_channels * 2),
            ConvBNReLU(decode_channels * 2, decode_channels)
        )
        
        # Segmentation head
        self.segmentation_head = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels),
            nn.Dropout2d(p=dropout, inplace=True),
            Conv(decode_channels, num_classes, kernel_size=1)
        )
        self.init_weight()

    def forward(self, x, h, w):
        """
        Args:
            x: (B, N, C) from ViT encoder
            h, w: target output height and width
        """
        B, N, C = x.shape
        
        # Reshape to spatial format
        H_feat = W_feat = int(math.sqrt(N))
        x = x.transpose(1, 2).reshape(B, C, H_feat, W_feat)
        
        # Multi-scale processing
        x4 = self.b4(self.pre_conv(x))  # Finest scale (4x channels)
        
        x3 = self.b3(self.up1(x4))      # Medium scale (2x channels)
        
        x2 = self.b2(self.up2(x3))      # Coarsest scale (1x channels)
        
        # Upsample all features to same spatial size for fusion
        x4_up = F.interpolate(x4, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x3_up = F.interpolate(x3, size=x2.shape[2:], mode='bilinear', align_corners=False)
        
        # Fuse multi-scale features (concatenate along channel dimension)
        x_fused = torch.cat([x4_up, x3_up, x2], dim=1)
        x_fused = self.fusion(x_fused)
        
        # Generate segmentation map
        x = self.segmentation_head(x_fused)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        return x

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Segmentation_ViT_UNetFormer(nn.Module):
    """
    ViT encoder + IMPROVED UNetFormer decoder.
    
    Combines Vision Transformer encoder with enhanced UNetFormer decoder
    featuring multi-scale feature processing for better object separation.
    """
    def __init__(
        self,
        encoder,
        encoder_output_dim,
        num_classes=1,
        decoder_channels=64,
        window_size=8,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_output_dim = encoder_output_dim
        self.num_classes = num_classes
        
        # Patch size
        self.patch_size = encoder.patch_embed.patch_size[0] if hasattr(encoder, 'patch_embed') else 16
        
        # IMPROVED UNetFormer decoder with multi-scale features
        self.decoder = UNetFormerDecoder(
            in_channels=encoder_output_dim,
            decode_channels=decoder_channels,
            dropout=dropout,
            window_size=window_size,
            num_classes=num_classes
        )
    
    @property
    def blocks(self):
        return self.encoder.blocks

    def no_weight_decay(self):
        return self.encoder.no_weight_decay()
    
    def forward(self, x):
        B, _, H, W = x.shape
        
        # Pass through encoder
        x_enc = self.encoder.patch_embed(x)
        
        # Add CLS token
        if hasattr(self.encoder, 'cls_token') and self.encoder.cls_token is not None:
            cls_tokens = self.encoder.cls_token.expand(B, -1, -1)
            x_enc = torch.cat((cls_tokens, x_enc), dim=1)
        
        # Add position embedding
        if hasattr(self.encoder, 'pos_embed') and self.encoder.pos_embed is not None:
            x_enc = x_enc + self.encoder.pos_embed
        
        if hasattr(self.encoder, 'pos_drop'):
            x_enc = self.encoder.pos_drop(x_enc)
        
        # Pass through transformer blocks
        for blk in self.encoder.blocks:
            x_enc = blk(x_enc)
        
        # Apply final norm
        if hasattr(self.encoder, 'norm'):
            x_enc = self.encoder.norm(x_enc)
        
        # Remove CLS token
        H_feat = H // self.patch_size
        W_feat = W // self.patch_size
        if x_enc.shape[1] == H_feat * W_feat + 1:
            x_enc = x_enc[:, 1:]  # Remove CLS token
        
        # Pass through IMPROVED decoder
        masks = self.decoder(x_enc, H, W)
        
        return masks