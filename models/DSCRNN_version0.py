import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size, padding=1):
        super(DepthwiseConv1D, self).__init__()
        self.depthwise = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=kernel_size, 
            padding=padding,
            groups=in_channels
        )

    def forward(self, x):
        return self.depthwise(x)


class DCRNNModel_Version0(nn.Module):
    '''
    Depthwise Separable Conv + RNN Model with Proxy Learning
    '''
    def __init__(self, in_channel=1, num_classes=5, lstm_hidden=45, lstm_layers=3):
        super(DCRNNModel_Version0, self).__init__()

        # SeparableConv layers
        self.depthwise1 = DepthwiseConv1D(in_channels=in_channel, kernel_size=15, padding=7)
        self.pointwise1 = nn.Conv1d(in_channels=in_channel, out_channels=180, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(180)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=90)

        self.depthwise2 = DepthwiseConv1D(in_channels=90, kernel_size=7, padding=3)
        self.pointwise2 = nn.Conv1d(in_channels=90, out_channels=90, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(90)
        self.maxpool2 = nn.AdaptiveMaxPool1d(output_size=45)

        self.depthwise3 = DepthwiseConv1D(in_channels=45, kernel_size=5, padding=2)
        self.pointwise3 = nn.Conv1d(in_channels=45, out_channels=45, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(45)
        

        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=180, 
            hidden_size=lstm_hidden, 
            num_layers=lstm_layers, 
            batch_first=True, 
            bidirectional=True
        )
        
        # LSTM
        lstm_output_size = lstm_hidden * 2


        self.fc = nn.Linear(lstm_output_size, num_classes)
        

        self.proxies = nn.Parameter(torch.randn(num_classes, lstm_output_size))
        nn.init.kaiming_normal_(self.proxies)

    def forward(self, x, return_features: bool = True):
        # x shape: (batch, in_channel, seq_len)
        
        # CNN layers
        x = self.depthwise1(x)
        x = self.pointwise1(x)
        x = self.bn1(x)
        x = F.relu(x)  

        x = self.depthwise2(x)
        x = self.pointwise2(x)
        x = self.bn2(x) 
        x = F.relu(x)
        x = self.maxpool2(x)
        # x shape: (batch, 180, seq_len)

        # CNN features
        x = self.depthwise3(x)
        x = self.pointwise3(x)
        x = self.bn3(x)
        x = F.relu(x)


        
        # LSTM
        x = x.permute(0, 2, 1)  # (batch, seq_len, 180)
        lstm_out, (h_n, c_n) = self.lstm(x)
      

   
        
        # Classification head
        out = self.fc(lstm_out)  # (batch, num_classes)
        
        if return_features:
            return out, lstm_out  # features는 (batch, 80)
        return out, None

    def get_proxies(self) -> torch.Tensor:
        """학습된 proxy 벡터 반환 - shape: (num_classes, 80)"""
        return self.proxies


# 테스트 코드
