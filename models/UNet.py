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

# 이제 가능
from loss_functions import *  

'''====================================================================== ResUNet Main Class ======================================================================'''


class UNet(nn.Module):
    def __init__(self, nOUT, in_channels=1, out_ch=180, mid_ch=30,
                 inconv_size=5, rub0_layers=6, use_proxy=True, verbose=False):
        super().__init__()
        
        self.nOUT = nOUT
        self.out_ch = out_ch
        self.use_proxy = use_proxy
        self.verbose = verbose
        
        if self.verbose:
            print(f"[ResUNet] Initializing: nOUT={nOUT}, out_ch={out_ch}, use_proxy={use_proxy}")
      
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



        x = F.leaky_relu(self.bn1(self.conv1(x)))  # -> [B, C, T']

        x = self.rub_0(x)


        x = F.dropout(x, p=0.5, training=self.training)

        features = self.maxpool(x)



        logits = self.fc_1(features)

        if return_features: 
            return logits, features
        return logits
    
    def get_proxies(self):
        return self.proxies


'''====================================================================== UNet Modules ======================================================================'''


class ResidualUBlock(nn.Module):
    def __init__(self, out_ch, mid_ch, layers, downsampling=True, verbose=False):
        super().__init__()
        self.verbose = verbose

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
        if self.verbose: print(f"[RUB] Input: {x.shape}")
        x_in = F.leaky_relu(self.bn1(self.conv1(x)))
        if self.verbose: print(f"[RUB] After Conv1+BN+ReLU: {x_in.shape}")

        out = x_in
        encoder_out = []
        for i, enc in enumerate(self.encoders):
            out = enc(out)
            encoder_out.append(out)
            if self.verbose: print(f"[RUB] Encoder {i}: {out.shape}")

        out = self.bottleneck(out)
        if self.verbose: print(f"[RUB] Bottleneck: {out.shape}")

        for idx, dec in enumerate(self.decoders):
            skip = encoder_out[-1 - idx]
            if out.size(-1) != skip.size(-1):
                out = F.interpolate(out, size=skip.size(-1), mode='linear', align_corners=False)
            out = dec(torch.cat([out, skip], dim=1))
            if self.verbose: print(f"[RUB] Decoder {idx}: {out.shape}")

        if out.size(-1) != x_in.size(-1):
            out = out[..., :x_in.size(-1)]
            if self.verbose: print(f"[RUB] Cropped to match input: {out.shape}")

        out += x_in
        if self.verbose: print(f"[RUB] After Residual Add: {out.shape}")

        if self.downsample:
            out = self.idfunc_0(out)
            out = self.idfunc_1(out)
            if self.verbose: print(f"[RUB] After Downsampling: {out.shape}")

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


'''====================================================================== ResUNet Debug ======================================================================'''

from typing import Tuple, List

def calculate_conv1d_output_size(input_size: int, kernel_size: int, stride: int = 1, 
                                padding: int = 0, dilation: int = 1) -> int:
    """Conv1D 출력 크기 계산"""
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

def calculate_convtranspose1d_output_size(input_size: int, kernel_size: int, stride: int = 1,
                                        padding: int = 0, output_padding: int = 0, dilation: int = 1) -> int:
    """ConvTranspose1D 출력 크기 계산"""
    return (input_size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

def calculate_pool_output_size(input_size: int, kernel_size: int, stride: int = None) -> int:
    """Pooling 출력 크기 계산"""
    if stride is None:
        stride = kernel_size
    return (input_size - kernel_size) // stride + 1

class DimensionTracker:
    """차원 변화를 추적하는 클래스"""
    def __init__(self):
        self.history = []
    
    def add_step(self, layer_name: str, input_shape: Tuple, output_shape: Tuple, operation_detail: str = ""):
        self.history.append({
            'layer': layer_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'detail': operation_detail
        })
    
    def print_summary(self):
        print("\n" + "="*100)
        print("                                    차원 변화 요약")
        print("="*100)
        print(f"{'Layer':<25} {'Input Shape':<20} {'Output Shape':<20} {'Detail':<30}")
        print("-"*100)
        
        for step in self.history:
            input_str = str(step['input_shape'])
            output_str = str(step['output_shape'])
            print(f"{step['layer']:<25} {input_str:<20} {output_str:<20} {step['detail']:<30}")
        
        print("="*100)

def debug_resunet_dimensions(nOUT: int = 5, in_channels: int = 1, out_ch: int = 180, 
                           mid_ch: int = 30, inconv_size: int = 15, rub0_layers: int = 3,
                           input_length: int = 2500):
    """
    ResUNet 모델의 차원 변화를 계산하고 출력
    
    Args:
        nOUT: 출력 클래스 수
        in_channels: 입력 채널 수
        out_ch: 출력 채널 수
        mid_ch: 중간 채널 수
        inconv_size: 초기 conv 커널 크기
        rub0_layers: ResidualUBlock 레이어 수
        input_length: 입력 시퀀스 길이
    """
    
    tracker = DimensionTracker()
    
    print(f"\n{'='*60}")
    print(f"ResUNet 차원 계산 (Input Length: {input_length})")
    print(f"{'='*60}")
    
    # 초기 입력
    current_shape = (1, in_channels, input_length)  # (B, C, L)
    print(f"초기 입력: {current_shape}")
    
    # 1. 초기 Conv1D
    # conv = nn.Conv1d(in_channels, out_ch, kernel_size=inconv_size, padding=7, stride=2, bias=False)
    padding = 7
    stride = 2
    new_length = calculate_conv1d_output_size(input_length, inconv_size, stride, padding)
    new_shape = (1, out_ch, new_length)
    
    tracker.add_step("Initial Conv1D", current_shape, new_shape, 
                    f"k={inconv_size}, s={stride}, p={padding}")
    current_shape = new_shape
    
    # 2. ResidualUBlock 계산
    print(f"\n--- ResidualUBlock 계산 (layers={rub0_layers}) ---")
    
    # RUB 내부 계산
    rub_input_shape = current_shape
    rub_length = current_shape[2]
    
    # Conv1 in RUB (K=9, P=4)
    K, P = 9, 4
    conv1_length = calculate_conv1d_output_size(rub_length, K, 1, P)
    tracker.add_step("RUB Conv1", rub_input_shape, (1, out_ch, conv1_length),
                    f"k={K}, s=1, p={P}")
    
    # Encoder 단계들
    encoder_lengths = []
    current_length = conv1_length
    current_channels = out_ch
    
    for i in range(rub0_layers):
        # Encoder
        if i == 0:
            in_ch = out_ch
        else:
            in_ch = mid_ch
        
        # stride=2로 downsampling
        new_length = calculate_conv1d_output_size(current_length, K, 2, P)
        encoder_lengths.append(new_length)
        
        tracker.add_step(f"RUB Encoder {i}", (1, in_ch, current_length), 
                        (1, mid_ch, new_length), f"k={K}, s=2, p={P}")
        
        current_length = new_length
        current_channels = mid_ch
    
    # Bottleneck
    bottleneck_length = calculate_conv1d_output_size(current_length, K, 1, P)
    tracker.add_step("RUB Bottleneck", (1, mid_ch, current_length), 
                     (1, mid_ch, bottleneck_length), f"k={K}, s=1, p={P}")
    current_length = bottleneck_length
    
    # Decoder 단계들
    for i in range(rub0_layers):
        skip_length = encoder_lengths[-1-i]
        
        # ConvTranspose1d: stride=2, output_padding=1
        new_length = calculate_convtranspose1d_output_size(current_length, K, 2, P, 1)
        
        # Skip connection을 위한 크기 조정 고려
        if new_length != skip_length:
            new_length = skip_length  # interpolation으로 맞춤
        
        if i == rub0_layers - 1:
            out_ch_dec = out_ch
        else:
            out_ch_dec = mid_ch
            
        tracker.add_step(f"RUB Decoder {i}", (1, mid_ch*2, current_length), 
                        (1, out_ch_dec, new_length), 
                        f"ConvT k={K}, s=2, p={P}, op=1 + skip")
        current_length = new_length
    
    # Residual connection 후 크기 조정
    if current_length != conv1_length:
        current_length = conv1_length  # crop to match
        tracker.add_step("RUB Crop", (1, out_ch, current_length), 
                        (1, out_ch, conv1_length), "crop to match input")
    
    # Downsampling (if enabled)
    # self.idfunc_0 = nn.AvgPool1d(kernel_size=2, stride=2)
    downsampled_length = calculate_pool_output_size(conv1_length, 2, 2)
    final_rub_shape = (1, out_ch, downsampled_length)
    
    tracker.add_step("RUB Downsample", (1, out_ch, conv1_length), final_rub_shape,
                    "AvgPool1d k=2, s=2 + Conv1d k=1")
    
    current_shape = final_rub_shape
    
    # 3. Global Pooling
    # nn.AdaptiveMaxPool1d(output_size=1)
    pooled_shape = (1, out_ch, 1)
    tracker.add_step("AdaptiveMaxPool1d", current_shape, pooled_shape, "output_size=1")
    
    # 4. Squeeze for FC
    features_shape = (out_ch,)
    tracker.add_step("Squeeze", pooled_shape, features_shape, "remove spatial dim")
    
    # 5. Fully Connected Layers
    # Linear(out_ch, out_ch // 2)
    fc1_shape = (out_ch // 2,)
    tracker.add_step("FC1", features_shape, fc1_shape, f"Linear {out_ch} -> {out_ch//2}")
    
    # Linear(out_ch // 2, nOUT)
    final_shape = (nOUT,)
    tracker.add_step("FC2", fc1_shape, final_shape, f"Linear {out_ch//2} -> {nOUT}")
    
    # 요약 출력
    tracker.print_summary()
    
    print(f"\n최종 출력 차원: {final_shape}")
    print(f"Features 차원: {features_shape}")
    
    if hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
        print(f"\n예상 GPU 메모리 사용량 (batch_size=1):")
        # 대략적인 메모리 계산 (float32 기준)
        max_intermediate = max([
            shape[1] * shape[2] if len(shape) == 3 else shape[0] 
            for _, shape in [(step['input_shape'], step['output_shape']) for step in tracker.history]
            for shape in [_[0], _[1]]
        ])
        memory_mb = max_intermediate * 4 / (1024**2)  # 4 bytes per float32
        print(f"  중간 활성화 최대 크기: ~{memory_mb:.2f} MB")

def test_with_actual_model():
    """실제 모델로 차원 검증"""
    print("\n" + "="*60)
    print("실제 모델로 차원 검증")
    print("="*60)
    
    try:
        # 실제 모델 생성 (원본 코드가 있다고 가정)
        model = ResUNet(
            nOUT=5,
            in_channels=1,
            out_ch=180,
            mid_ch=30,
            inconv_size=5,
            rub0_layers=6,
            use_proxy=True,
            verbose=False
        )
        
        model.eval()
        
        # 테스트 입력
        test_input = torch.randn(1, 1, 1800)
        
        with torch.no_grad():
            logits, features = model(test_input, return_features=True)
        
        print(f"실제 모델 결과:")
        print(f"  입력 차원: {test_input.shape}")
        print(f"  로짓 차원: {logits.shape}")
        print(f"  특성 차원: {features.shape}")
        
        if hasattr(model, 'proxies'):
            print(f"  프록시 차원: {model.proxies.shape}")
            
    except Exception as e:
        print(f"실제 모델 테스트 실패: {e}")
        print("모델 클래스가 정의되지 않았거나 의존성이 누락되었을 수 있습니다.")

if __name__ == "__main__":
    # 다양한 설정으로 테스트
    configs = [
        {
            'name': 'Default Config',
            'nOUT': 5,
            'in_channels': 1,
            'out_ch': 180,
            'mid_ch': 30,
            'inconv_size': 5,
            'rub0_layers': 6,
            'input_length': 1800
        },
        {
            'name': 'Large Config',
            'nOUT': 10,
            'in_channels': 1,
            'out_ch': 256,
            'mid_ch': 64,
            'inconv_size': 15,
            'rub0_layers': 6,
            'input_length': 5000
        }
    ]
    
    for config in configs:
        print(f"\n{'#'*80}")
        print(f"테스트: {config['name']}")
        print(f"{'#'*80}")
        
        debug_resunet_dimensions(
            nOUT=config['nOUT'],
            in_channels=config['in_channels'],
            out_ch=config['out_ch'],
            mid_ch=config['mid_ch'],
            inconv_size=config['inconv_size'],
            rub0_layers=config['rub0_layers'],
            input_length=config['input_length']
        )
    
    # 실제 모델 테스트 시도
    test_with_actual_model()




