import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=filters, 
            kernel_size=kernel_size, 
            padding=padding
        )
        self.pool = nn.AvgPool1d(kernel_size=2)
        self.bn = nn.BatchNorm1d(filters)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x), negative_slope=0.01)
        x = self.pool(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class HUnivModel(nn.Module):
    def __init__(self, nOUT, in_channels=1):
        super().__init__()
        
        # 논문에 따른 정확한 구조
        self.conv1 = ConvBlock(in_channels=in_channels, filters=32, kernel_size=8)
        self.conv2 = ConvBlock(in_channels=32, filters=48, kernel_size=12)
        self.conv3 = ConvBlock(in_channels=48, filters=64, kernel_size=16)
        self.conv4 = ConvBlock(in_channels=64, filters=96, kernel_size=20)
        self.conv5 = ConvBlock(in_channels=96, filters=128, kernel_size=24)
        self.conv6 = ConvBlock(in_channels=128, filters=256, kernel_size=26)
        self.conv7 = ConvBlock(in_channels=256, filters=384, kernel_size=28)
        
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.dense = nn.Linear(384, nOUT)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        features = self.pool(x)
        features = features.squeeze(2)
        x = self.dense(features)

        
        return x, features


def l1_regularization(model, lambda_l1=0.001):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.abs(param).sum()
    return lambda_l1 * l1_loss

'''
L2, L1 regularization
===============================================
model = HUnivModel(nOUT=8, in_channels=1)
optimizer = torch.optim.Adam(
    model.parameters(), 
    lr=0.0005, 
    weight_decay=0.002  # L2 regularization
)

 l1_loss = l1_regularization(model, lambda_l1=0.001)
로 penelty 주면됨
'''
