"""
DSCRNN: Depthwise Separable Convolution + RNN
ECG 분류를 위한 경량 딥러닝 모델


"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv1D(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, input_dim, output_dim, kernel_size=3, padding=1):
        super(DepthwiseSeparableConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            input_dim, input_dim, 
            kernel_size=kernel_size, 
            padding=padding,
            groups=input_dim,
            bias=False
        )
        self.pointwise = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        # x = self.relu(x)
        x = self.bn(x)
        x = F.relu(x)
        return x



class DCRNNModel_backbone(nn.Module):

    def __init__(self, in_channel=1, num_classes=5,
                 conv_channels=[64, 128, 256],
                 lstm_hidden=128, lstm_layers=2):
        super(DCRNNModel_backbone, self).__init__()
        
        self.num_conv_layers = len(conv_channels)
        
        # Conv1 (항상 사용)
        self.conv1 = DepthwiseSeparableConv1D(in_channel, conv_channels[0], kernel_size=15, padding=7)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # Conv2 (3-layer 모델에만 사용)
        if self.num_conv_layers == 3:
            self.conv2 = DepthwiseSeparableConv1D(conv_channels[0], conv_channels[1], kernel_size=5, padding=2)
            self.pool2 = nn.MaxPool1d(kernel_size=2)
            self.conv3 = DepthwiseSeparableConv1D(conv_channels[1], conv_channels[2], kernel_size=5, padding=2)
        elif self.num_conv_layers == 2:
            self.conv3 = DepthwiseSeparableConv1D(conv_channels[0], conv_channels[1], kernel_size=5, padding=2)
        # LSTM
        self.lstm = nn.LSTM(
            input_size=conv_channels[-1],  # 마지막 conv의 출력 채널
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.2 if lstm_layers > 1 else 0
        )
        
        # Feature dimension
        self.feature_dim = lstm_hidden  # Bidirectional
        
        # Fully connected
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Proxy vectors
        self.proxies = nn.Parameter(torch.randn(num_classes, self.feature_dim))
        nn.init.kaiming_normal_(self.proxies)
    
    def forward(self, x, return_features=True):
        """
        Args:
            x: [B, C, L] - Batch, Channel, Length
            return_features: feature 벡터 반환 여부
        
        Returns:
            logits: [B, num_classes]
            features: [B, feature_dim] (if return_features=True)
        """
        # Conv1 + Pool1 (항상 실행)
        x = self.conv1(x)
        x = self.pool1(x)
        
        # Conv2 + Pool2 (3-layer 모델에만 실행)
        if self.num_conv_layers == 3:
            x = self.conv2(x)
            x = self.pool2(x)
        
        # Conv3 (항상 실행)
        x = self.conv3(x)
 
        # LSTM
        x = x.permute(0, 2, 1)  # [B, L', conv_channels[-1]]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Last time step
        features = lstm_out[:, -1, :]  # [B, lstm_hidden*2]
        
        # Classification
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits, None
    
    def get_proxies(self):
        """Proxy 벡터 반환"""
        return self.proxies


# ===================== 버전별 모델 (논문 기반) =====================

def DCRNNModel(in_channel=1, num_classes=5):
    """
    Tiny 버전 (depth 줄임):
    - Conv1: 1→256 (kernel 128) + MaxPool
    - Conv3: 256→128 (kernel 64) 
    - LSTM: hidden 50, layers 1
    """
    return DCRNNModel_backbone(
        in_channel=in_channel,
        num_classes=num_classes,
        conv_channels=[32, 64],  # 2-layer만 사용
        lstm_hidden=50,
        lstm_layers=1
    )


