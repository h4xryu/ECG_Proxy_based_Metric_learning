import torch


def count_parameters_by_module(model):
    """
    모델 내 각 모듈별 파라미터 수 계산 및 출력
    
    Args:
        model: PyTorch 모델
    
    Returns:
        dict: 모듈명을 key로 하는 파라미터 수 딕셔너리
    """
    result = {}
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                result[name] = param_count

    total_params = 0
    print("=== 모듈별 파라미터 수 ===")
    for module_name in sorted(result.keys()):
        param_count = result[module_name]
        total_params += param_count

    print(f"\n총 파라미터 수: {total_params:,} ({total_params / 1e6:.2f}M)")
    return result


def optimizer_to(optim, device):
    """
    optimizer의 state를 특정 디바이스로 이동
    주로 체크포인트 로드 시 사용
    
    Args:
        optim: PyTorch optimizer
        device: 목표 디바이스 (cuda 또는 cpu)
    """
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
