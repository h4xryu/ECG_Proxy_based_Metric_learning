"""
SE-ResNet + LSTM for 1D ECG Classification
Squeeze-and-Excitation ResNet with Bottleneck blocks + LSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SELayer(nn.Module):
    """Squeeze-and-Excitation Layer"""
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)



class Bottleneck(nn.Module):
    """
    SE-ResNet Bottleneck Block for 1D
    expansion = 4 (기본값, 조정 가능)
    """
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16, expansion=None):
        super(Bottleneck, self).__init__()
        
        # expansion 동적 설정
        if expansion is not None:
            self.expansion = expansion
        
        # 1x1 conv
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        # 3x3 conv
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        # 1x1 conv (expansion)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        
        # SE layer
        self.se = SELayer(planes * self.expansion, reduction)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        # SE block
        out = self.se(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class SEResNetLSTM_backbone(nn.Module):

    def __init__(self, block, layers, in_channel=1, num_classes=5,
                 lstm_hidden=128, lstm_layers=2, reduction=16):
        super(SEResNetLSTM_backbone, self).__init__()
        self.inplanes = 64
        self.reduction = reduction
        
        # Initial layers
        self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # LSTM의 input_size는 layer3의 출력 채널 수
        lstm_input_size = 128 * block.expansion  # 256 * 4 = 1024
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.2 if lstm_layers > 1 else 0   
        )
        
        # Feature dimension
        self.feature_dim = lstm_hidden  # Bidirectional
        
        # Classification head
        self.fc = nn.Linear(self.feature_dim, num_classes)
        
        # Proxy vectors
        self.proxies = nn.Parameter(torch.randn(num_classes, self.feature_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _make_layer(self, block, planes, blocks, stride=1):
        """ResNet layer 생성"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=self.reduction))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, return_features=True):

        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # SE-ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        
        # LSTM
        x = x.permute(0, 2, 1)  # [B, L', C]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Last time step
        features = lstm_out[:, -1, :]  # [B, lstm_hidden*2]
        
        # Classification
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        return logits
    
    def get_proxies(self):
        """Proxy 벡터 반환"""
        return self.proxies


# ===================== 모델 버전 (ResNet 스타일) =====================

def SEResNetLSTM(in_channel=1, num_classes=5):

    return SEResNetLSTM_backbone(
        Bottleneck, [2, 2, 2],
        in_channel=in_channel,
        num_classes=num_classes,
        lstm_hidden=50,
        lstm_layers=1,
        reduction=16
    )


