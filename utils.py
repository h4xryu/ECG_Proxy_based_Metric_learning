"""
유틸리티 함수 - 3개 카테고리로 정리
1. 전처리 & 데이터
2. 체크포인트 관리  
3. 학습 설정 출력
"""
import os
import torch
import numpy as np
import wfdb
import json
from datetime import datetime
from collections import Counter
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
from torch.utils.data import DataLoader
from dataloader import ECGDataloader


"""
================================================================================
1. 전처리 & 데이터
================================================================================
"""

# AAMI 표준 레이블 매핑
LABEL_GROUP_MAP = {
    'N': 'N', 'L': 'N', 'R': 'N',  # Normal beats
    'V': 'V', 'E': 'V',             # Ventricular ectopic beats
    'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S', 'e': 'S', 'j': 'S',  # Supraventricular ectopic beats
    'F': 'F',                        # Fusion beats
    '/': 'Q', 'f': 'Q', 'Q': 'Q'    # Unknown beats
}

# MIT-BIH 데이터셋 분할 (inter-patient)
DS1_TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124',
    '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
]
DS2_TEST_RECORDS = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202',
    '210', '212', '213', '214', '219', '221', '222', '228', '231', '232', '233', '234'
]


def label2index(label):
    """ECG 레이블을 정수 인덱스로 변환 (N:0, S:1, V:2, F:3, Q:4)"""
    return {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}[label]




def normalize_beat(beat_signal):
    """개별 비트 Z-score 정규화"""
    mean = np.mean(beat_signal)
    std = np.std(beat_signal)
    return (beat_signal - mean) / (std + 1e-14)


def mkdir(path):
    """디렉토리 생성"""
    if not os.path.exists(path):
        os.makedirs(path)


def resample_signal(signal, fs_in, fs_out, target_length):
    """신호를 목표 길이로 리샘플링"""
    if fs_out == fs_in and len(signal) == target_length:
        return signal
    
    x_old = np.linspace(0, 1, num=len(signal))
    x_new = np.linspace(0, 1, num=target_length)
    return interp1d(x_old, signal)(x_new)


def apply_bandpass_filter(signal, fs=360, lowcut=0.1, highcut=100.0, filter_order=256):
    """FIR 밴드패스 필터 적용 (0.1-100Hz)"""
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    fir_coeff = firwin(filter_order + 1, [low, high], pass_zero=False)
    filtered_signal = filtfilt(fir_coeff, 1.0, signal)
    
    return filtered_signal


def preprocess_leads(sig_data, sig_names, leads, fs):
    """선택된 리드에 대해 밴드패스 필터 적용"""
    filtered_data = {}
    for lead in leads:
        if lead in sig_names:
            idx = sig_names.index(lead)
            raw_signal = sig_data[:, idx]
            filtered = apply_bandpass_filter(raw_signal, fs=fs, lowcut=0.1, highcut=100.0)
            filtered_data[lead] = filtered
    return filtered_data


def extract_single_beat(filtered_data, leads, start, end, fs, fs_out, target_length):
    """단일 비트를 추출하고 전처리"""
    segment = []
    expected_length = end - start
    
    for lead in leads:
        if lead not in filtered_data:
            return None
        
        lead_signal = filtered_data[lead][start:end]
        
        if len(lead_signal) != expected_length:
            return None
        
        # 리샘플링
        lead_signal = resample_signal(lead_signal, fs, fs_out, target_length)
        segment.append(lead_signal)
    
    if len(segment) != len(leads):
        return None
    
    # Single lead: (target_length,) 형태
    if len(leads) == 1:
        segment = segment[0].astype(np.float32)
        segment = normalize_beat(segment)
    # Multi lead: (num_leads, target_length) 형태
    else:
        segment = np.stack(segment).astype(np.float32)
        for i in range(len(leads)):
            segment[i] = normalize_beat(segment[i])
    
    return segment


def extract_beats(record_path, record_name, leads, fs_out=360, segment_seconds=1.0):
    """MIT-BIH 레코드에서 비트 추출 및 전처리"""
    # 어노테이션 및 신호 데이터 로드
    ann_data = wfdb.rdann(os.path.join(record_path, record_name), 'atr')
    sig_data, meta = wfdb.rdsamp(os.path.join(record_path, record_name))
    fs = meta['fs']
    sig_names = meta['sig_name']
    
    # 리드별 필터링
    filtered_data = preprocess_leads(sig_data, sig_names, leads, fs)
    
    # 추출 파라미터 계산
    target_length = int(segment_seconds * fs_out)
    half_segment = int(segment_seconds * fs / 2)
    
    beats, labels, groups = [], [], []
    
    # R-peak 위치별 비트 추출
    for pos, symbol in zip(ann_data.sample, ann_data.symbol):
        if symbol not in LABEL_GROUP_MAP:
            continue
        
        start = pos - half_segment
        end = pos + half_segment
        
        # 경계 체크
        if start < 0 or end > len(sig_data):
            continue
        
        # 비트 추출 및 전처리
        segment = extract_single_beat(
            filtered_data, leads, start, end, fs, fs_out, target_length
        )
        
        if segment is not None:
            beats.append(segment)
            labels.append(symbol)
            groups.append(LABEL_GROUP_MAP[symbol])
    
    return beats, labels, groups


def process_records(data_path, record_names, leads, fs_out, segment_seconds, desc="Processing"):
    """여러 레코드에서 비트 추출"""
    all_beats, all_labels, all_groups = [], [], []
    
    for record_name in tqdm(record_names, desc=desc):
        try:
            beats, labels, groups = extract_beats(
                data_path, record_name, leads, fs_out, segment_seconds
            )
            all_beats.extend(beats)
            all_labels.extend(labels)
            all_groups.extend(groups)
        except Exception as e:
            print(f"Warning: Error processing {record_name}: {e}")
            continue
    
    return all_beats, all_labels, all_groups


def get_cache_dir(base_output_dir, segment_seconds, fs_out, leads):
    """캐시 디렉토리 경로 생성"""
    cache_key = f"seg{segment_seconds}_fs{fs_out}_{'_'.join(leads)}"
    return os.path.join(base_output_dir, cache_key)


def preprocess_and_cache_mitbih(data_path, output_base_dir, segment_seconds, fs_out, leads):
    """
    MIT-BIH 데이터를 전처리하고 캐싱
    이미 전처리된 데이터가 있으면 로드, 없으면 전처리 후 저장
    """
    cache_dir = get_cache_dir(output_base_dir, segment_seconds, fs_out, leads)
    
    train_data_path = os.path.join(cache_dir, 'train', 'train_data.npy')
    train_labels_path = os.path.join(cache_dir, 'train', 'train_labels.npy')
    test_data_path = os.path.join(cache_dir, 'test', 'test_data.npy')
    test_labels_path = os.path.join(cache_dir, 'test', 'test_labels.npy')
    
    # 캐시된 데이터가 있으면 바로 반환
    if all(os.path.exists(p) for p in [train_data_path, train_labels_path, test_data_path, test_labels_path]):
        print(f"✓ 캐시된 데이터를 로드합니다: {cache_dir}")
        return cache_dir
    
    # 전처리 수행
    print(f"\n{'='*70}")
    print(f"MIT-BIH 데이터 전처리 시작")
    print(f"{'='*70}")
    print(f"설정: segment={segment_seconds}s, fs={fs_out}Hz, leads={leads}")
    print(f"출력: {cache_dir}\n")
    
    # DS1 학습 데이터 처리
    train_beats, train_labels, train_groups = process_records(
        data_path, DS1_TRAIN_RECORDS, leads, fs_out, segment_seconds,
        desc="DS1 Train"
    )
    
    # DS2 테스트 데이터 처리
    test_beats, test_labels, test_groups = process_records(
        data_path, DS2_TEST_RECORDS, leads, fs_out, segment_seconds,
        desc="DS2 Test"
    )
    
    # 데이터 저장
    os.makedirs(os.path.join(cache_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(cache_dir, 'test'), exist_ok=True)
    
    X_train = np.array(train_beats, dtype=np.float32)
    y_train = np.array(train_groups)
    X_test = np.array(test_beats, dtype=np.float32)
    y_test = np.array(test_groups)
    
    np.save(train_data_path, X_train)
    np.save(train_labels_path, y_train)
    np.save(test_data_path, X_test)
    np.save(test_labels_path, y_test)
    
    # 메타데이터 저장
    metadata = {
        'segment_seconds': segment_seconds,
        'fs_output': fs_out,
        'leads': leads,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_class_distribution': dict(Counter(y_train)),
        'test_class_distribution': dict(Counter(y_test))
    }
    
    with open(os.path.join(cache_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ 전처리 완료!")
    print(f"  Train: {len(X_train)} samples - {dict(Counter(y_train))}")
    print(f"  Test: {len(X_test)} samples - {dict(Counter(y_test))}")
    print(f"{'='*70}\n")
    
    return cache_dir


def load_train_data(batch_size, num_workers=0):
    """
    학습 데이터 로드 (train.py 전역변수 사용)
    전처리가 안되어 있으면 자동으로 전처리 후 캐싱
    """
    from train import (path_mitbih_raw, path_processed_base, 
                       segment_seconds, fs_out, ecg_leads)
    
    # 전처리 및 캐싱
    cache_dir = preprocess_and_cache_mitbih(
        path_mitbih_raw, path_processed_base, 
        segment_seconds, fs_out, ecg_leads
    )
    
    # 데이터 로드
    train_data = np.load(os.path.join(cache_dir, 'train', 'train_data.npy'))
    train_labels = np.load(os.path.join(cache_dir, 'train', 'train_labels.npy'))
    
    # 레이블을 인덱스로 변환
    train_labels = np.array([label2index(i) for i in train_labels])
    
    # 채널 차원 추가 (single lead인 경우)
    if train_data.ndim == 2:
        train_data = np.expand_dims(train_data, axis=1)
    
    train_loader = DataLoader(
        ECGDataloader(train_data, train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    return train_loader


def load_test_data(batch_size, num_workers=0):
    """
    테스트 데이터 로드 (train.py 전역변수 사용)
    전처리가 안되어 있으면 자동으로 전처리 후 캐싱
    """
    from train import (path_mitbih_raw, path_processed_base, 
                       segment_seconds, fs_out, ecg_leads)
    
    # 전처리 및 캐싱
    cache_dir = preprocess_and_cache_mitbih(
        path_mitbih_raw, path_processed_base, 
        segment_seconds, fs_out, ecg_leads
    )
    
    # 데이터 로드
    test_data = np.load(os.path.join(cache_dir, 'test', 'test_data.npy'))
    test_labels = np.load(os.path.join(cache_dir, 'test', 'test_labels.npy'))
    
    # 레이블을 인덱스로 변환
    test_labels = np.array([label2index(i) for i in test_labels])
    
    # 채널 차원 추가 (single lead인 경우)
    if test_data.ndim == 2:
        test_data = np.expand_dims(test_data, axis=1)
    
    test_loader = DataLoader(
        ECGDataloader(test_data, test_labels),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return test_loader


"""
================================================================================
2. 체크포인트 관리
================================================================================
"""

def create_log_directory():
    """타임스탬프 기반 로그 디렉토리 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.getcwd(), 'logs', f'{timestamp}')
    
    os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
    
    return os.path.join(log_dir, 'tensorboard')


def get_model_dir(log_dir):
    """모델 저장 디렉토리 경로"""
    return os.path.join(
        os.path.dirname(log_dir.replace('tensorboard', 'checkpoints')), 
        'checkpoints'
    )


def create_checkpoint(model, optimizer, scheduler, epoch, f1_score, best_f1, class_names):
    """체크포인트 딕셔너리 생성"""
    return {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'f1_score': f1_score,
        'best_f1': best_f1,
        'class_names': class_names
    }


def save_checkpoint(checkpoint, model_dir, epoch, is_best):
    """체크포인트 저장"""
    torch.save(checkpoint, os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth'))
    
    if is_best:
        best_path = os.path.join(model_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        return best_path
    
    return None


def load_best_model(model, best_model_path, device):
    """최고 성능 모델 로드"""
    print(f"Loading best model from {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])


"""
================================================================================
3. 학습 설정 출력
================================================================================
"""

def print_model_info(model, model_name="Model"):
    """모델 정보 출력"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    print(f"Total Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    print(f"{'='*70}\n")


def print_training_config():
    """학습 설정 출력"""
    from train import (proxy_type, proxy_alpha, proxy_delta,
                       lambda_combined, batch_size, lr_initial, nepoch, device,
                       segment_seconds, fs_out, ecg_leads)
    
    print("\n" + "="*70)
    print("TRAINING CONFIGURATION")
    print("="*70)
    
    # 데이터 설정
    print(f"\n[Data Preprocessing]")
    print(f"  Segment Length: {segment_seconds}s ({int(segment_seconds * fs_out)} samples)")
    print(f"  Sampling Rate: {fs_out} Hz")
    print(f"  ECG Leads: {', '.join(ecg_leads)}")
    
    # 손실 함수 설정
    print(f"\n[Loss Function]")
    print(f"  λ (lambda): {lambda_combined:.2f}")
    
    if lambda_combined == 1.0:
        print(f"  Mode: CE only")
    elif lambda_combined == 0.0:
        print(f"  Mode: Proxy only")
        print(f"  Proxy Type: {proxy_type}")
        print(f"  Proxy α (alpha): {proxy_alpha}")
        print(f"  Proxy δ (delta): {proxy_delta}")
    else:
        print(f"  Mode: Combined")
        print(f"    → CE weight: {lambda_combined:.2f}")
        print(f"    → Proxy weight: {1-lambda_combined:.2f}")
        print(f"  Proxy Type: {proxy_type}")
        print(f"  Proxy α (alpha): {proxy_alpha}")
        print(f"  Proxy δ (delta): {proxy_delta}")
    
    # 학습 설정
    print(f"\n[Training]")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr_initial}")
    print(f"  Epochs: {nepoch}")
    print(f"  Device: {device}")
    
    print("="*70 + "\n")
