"""
UNet 기반 ECG 심전도 분류 모델
Proxy Loss를 활용한 Metric Learning 지원
"""
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 프로젝트 루트 경로를 sys.path에 추가
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)  

class UNet(nn.Module):

    
    def __init__(
        self,
        nOUT: int,
        in_channels: int = 1,
        out_ch: int = 180,
        mid_ch: int = 30,
        inconv_size: int = 5,
        rub0_layers: int = 6,
        use_proxy: bool = True
    ):

        super().__init__()
        
        self.nOUT = nOUT
        self.out_ch = out_ch
        self.use_proxy = use_proxy
        
        # 초기 컨볼루션 레이어
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_ch,
            kernel_size=inconv_size,
            padding=7, 
            stride=2, 
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        
        # Residual U-Block
        self.rub_0 = ResidualUBlock(out_ch, mid_ch, layers=rub0_layers)

        self.attn = nn.MultiheadAttention(embed_dim=out_ch, num_heads=1)
        
        # Global pooling
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        
        # 분류 헤드
        self.fc_1 = nn.Sequential(
            nn.Linear(out_ch, out_ch // 2),
            nn.BatchNorm1d(out_ch // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_ch // 2, nOUT)
        )
        
        # Proxy 파라미터 초기화
        if self.use_proxy:
            self.proxies = nn.Parameter(torch.randn(nOUT, out_ch))
            nn.init.kaiming_normal_(self.proxies)

    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        # 초기 컨볼루션
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        
        # Residual U-Block 통과
        x = self.rub_0(x)

        x_attn = x.permute(0, 2, 1)

        x_attn = self.attn(x_attn, x_attn, x_attn)[0]

        x_attn = x_attn.permute(0, 2, 1)
        x = x + x_attn
        
        # Dropout
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Global pooling 및 특징 벡터 추출
        features = self.maxpool(x).squeeze(-1)  # [B, out_ch, 1] -> [B, out_ch]
        
        # 분류 로짓 계산
        logits = self.fc_1(features)
        
        if return_features:
            return logits, features
        return logits, None
    
    def get_proxies(self) -> torch.Tensor:
        """
        학습된 proxy 벡터 반환
        
        Returns:
            proxies: Proxy 벡터 [nOUT, out_ch]
            
        Raises:
            AttributeError: use_proxy=False로 초기화된 경우
        """

        return self.proxies



class ResidualUBlock(nn.Module):

    
    def __init__(
        self,
        out_ch: int,
        mid_ch: int,
        layers: int,
        downsampling: bool = True,
        verbose: bool = False
    ):
        """
        Args:
            out_ch: 출력 채널 수
            mid_ch: 중간 채널 수
            layers: 인코더-디코더 레이어 수
            downsampling: 최종 다운샘플링 수행 여부
            verbose: 디버깅용 shape 출력 여부
        """
        super().__init__()
        self.verbose = verbose
        
        # 커널 크기 및 패딩 설정
        K = 9  # 커널 크기
        P = (K - 1) // 2  # 패딩
        
        # 초기 컨볼루션
        self.conv1 = nn.Conv1d(out_ch, out_ch, kernel_size=K, padding=P, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        
        # 인코더 및 디코더 레이어 생성
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        
        for idx in range(layers):
            # 인코더: 다운샘플링
            in_ch = out_ch if idx == 0 else mid_ch
            self.encoders.append(nn.Sequential(
                nn.Conv1d(in_ch, mid_ch, kernel_size=K, stride=2, padding=P, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.LeakyReLU()
            ))
            
            # 디코더: 업샘플링
            out_ch_dec = out_ch if idx == layers - 1 else mid_ch
            self.decoders.append(nn.Sequential(
                nn.ConvTranspose1d(
                    mid_ch * 2,  # skip connection으로 인한 채널 2배
                    out_ch_dec,
                    kernel_size=K,
                    stride=2,
                    padding=P,
                    output_padding=1,
                    bias=False
                ),
                nn.BatchNorm1d(out_ch_dec),
                nn.LeakyReLU()
            ))
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv1d(mid_ch, mid_ch, kernel_size=K, padding=P, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.LeakyReLU()
        )
        
        # 다운샘플링 레이어
        self.downsample = downsampling
        if self.downsample:
            self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
            self.idfunc_1 = nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 초기 컨볼루션
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))

        # 인코더: 다운샘플링 및 skip connection 저장
        out = x_in
        encoder_out = []
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            encoder_out.append(out)

        
        # Bottleneck
        out = self.bottleneck(out)

        
        # 디코더: 업샘플링 및 skip connection 결합
        for idx, dec in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            
            # 크기 불일치 보정
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(
                    out, 
                    size=skip.size(-1), 
                    mode='linear', 
                    align_corners=False
                )
            
            # Skip connection 결합
            out = dec(torch.cat([out, skip], dim=1))

        # 입력과 출력 크기 맞추기
        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]

        
        # Residual connection
        out = out + x_in

        # 다운샘플링
        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)

        
        return out

def count_parameters_by_module(model: nn.Module) -> dict:
    """
    모델의 각 모듈별 파라미터 수를 계산하고 출력
    
    Args:
        model: 파라미터를 계산할 PyTorch 모델
        
    Returns:
        result: 모듈명을 키로, 파라미터 수를 값으로 가지는 딕셔너리
    """
    result = {}
    total_params = 0
    
    # 리프 모듈(자식이 없는 모듈)만 카운트
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                result[name] = param_count
                total_params += param_count
    
    # 결과 출력
    print("=== 모듈별 파라미터 수 ===")
    for k in sorted(result.keys()):
        print(f"{k:>30}: {result[k]:,} parameters")
    
    print(f"\n총 파라미터 수: {total_params:,} ({total_params / 1e6:.3f}M)")
    return result


