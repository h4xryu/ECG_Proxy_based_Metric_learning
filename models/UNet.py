import torch
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.nn.functional as functional
import sys
from typing import Optional
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)  # .../ECG_ResUNet_enhanced_proxy
sys.path.insert(0, PROJECT_ROOT)

from loss_functions import *  

'''====================================================================== ResUNet Main Class ======================================================================'''


class UNet(nn.Module):
    def __init__(self, nOUT, in_channels=1, out_ch=180, mid_ch=30,
                 inconv_size=5, rub0_layers=7, use_proxy=True):
        super().__init__()
        
        self.nOUT = nOUT
        self.out_ch = out_ch
        self.use_proxy = use_proxy

        
     
      
        self.conv1 = nn.Conv1d(in_channels, out_ch, kernel_size=inconv_size, padding=7, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=rub0_layers)

        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
       
        self.fc_1 = nn.Sequential(
            nn.Linear(out_ch, out_ch // 2),
            nn.BatchNorm1d(out_ch // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_ch // 2, nOUT)
        )


        # Add learnable proxy parameters if using proxy loss
        if self.use_proxy:
            self.proxies = nn.Parameter(torch.randn(nOUT, out_ch))
            nn.init.kaiming_normal_(self.proxies)


    def forward(self, x, return_features=True):



        x = F.leaky_relu(self.bn1(self.conv1(x)))  
        x = self.rub_0(x)


        x = F.dropout(x, p=0.5, training=self.training)

        features = self.maxpool(x).squeeze(2)


        logits = self.fc_1(features)

        if return_features: 
            return logits, features
        return logits
    
    def get_proxies(self):
        return self.proxies


'''====================================================================== UNet Modules ======================================================================'''


class ResidualUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True):
        super().__init__()

        K = 9
        P = (K - 1) // 2

        self.conv1 = nn.Conv1d(out_ch, out_ch, kernel_size=K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for idx in range(layers):
            in_ch = out_ch if idx == 0 else mid_ch
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, kernel_size=K, stride=2, padding=P, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.LeakyReLU()
            ))

            out_ch_dec = out_ch if idx == layers - 1 else mid_ch
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(mid_ch * 2, out_ch_dec, kernel_size=K, stride=2, padding=P, output_padding=1, bias=False),
                nn.BatchNorm1d(out_ch_dec),
                nn.LeakyReLU()
            ))

        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
       
        )

        self.downsample = downsampling
        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x):

        x_in = F.leaky_relu(self.bn1(self.conv1(x)))


        out = x_in
        encoder_out = []
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            encoder_out.append(out)


        out = self.bottleneck(out)


        for idx, dec in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(out, size=skip.size(-1), mode='linear', align_corners=False)
            out = dec(torch.cat([out, skip], dim=1))


        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]


        out += x_in


        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)
   

        return out

def count_parameters_by_module(model):
    result = {}
    total_params = 0

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                result[name] = param_count
                total_params += param_count

    print("=== 모듈별 파라미터 수 ===")
    for k in sorted(result.keys()):
        print(f"{k:>30}: {result[k]:,} parameters")

    print(f"\n총 파라미터 수: {total_params:,} ({total_params / 1e6:.3f}M)")
    return result

