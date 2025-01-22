import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        return self.dropout(F.relu(self.bn(self.conv(x))))

class UNet_large(nn.Module):
    def __init__(self, in_channels=7, out_channels=5, dropout_rate=0.2):
        super(UNet_large, self).__init__()
        
        # Encoder
        self.down1 = DownsampleBlock(in_channels, 32, dropout_rate)
        self.down2 = DownsampleBlock(32, 64, dropout_rate)
        self.down3 = DownsampleBlock(64, 128, dropout_rate)
        self.down4 = DownsampleBlock(128, 256, dropout_rate)

        self.res1 = ResBlock(256, 256, dropout_rate)
        self.res2 = ResBlock(256, 256, dropout_rate)
        
        # Decoder
        self.up1 = UpsampleBlock(256, 128, dropout_rate)
        self.up2 = UpsampleBlock(256, 64, dropout_rate)
        self.up3 = UpsampleBlock(128,32, dropout_rate)
        self.up4 = UpsampleBlock(64, 32, dropout_rate)
        
        # Final convolution
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        # Self-attention
        # sa = self.self_attention(d4)
        
        # Residual blocks
        r = self.res1(d4)
        r = self.res2(r)
        
        # Decoder
        u1 = self.up1(r)
        u1 = torch.cat([u1, d3], dim=1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim=1)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim=1)
        u4 = self.up4(u3)
        
        # Final convolution
        output = self.final_conv(u4)
        
        return output