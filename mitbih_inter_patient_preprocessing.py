import numpy as np
import os
import wfdb
import argparse
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy.signal import firwin, filtfilt
from collections import Counter
import json

"""
================================================================================
MIT-BIH INTER-PATIENT PREPROCESSING - MIT-BIH 데이터셋 전처리
================================================================================
"""

# AAMI 표준에 따른 레이블 그룹 매핑
# N: Normal, S: Supraventricular, V: Ventricular, F: Fusion, Q: Unknown
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




def resample_signal(signal, fs_in, fs_out, target_length):
    """
    신호를 목표 길이로 리샘플링
    
    Args:
        signal: 입력 신호
        fs_in: 입력 샘플링 주파수
        fs_out: 출력 샘플링 주파수
        target_length: 목표 샘플 개수
    
    Returns:
        리샘플링된 신호
    """
    if fs_out == fs_in and len(signal) == target_length:
        return signal
    
    x_old = np.linspace(0, 1, num=len(signal))
    x_new = np.linspace(0, 1, num=target_length)
    return interp1d(x_old, signal)(x_new)


def apply_bandpass_filter(signal, fs=360, lowcut=0.1, highcut=100.0, filter_order=256):
    """
    FIR 밴드패스 필터 적용
    기본값: 0.1-100Hz (baseline wander 및 고주파 노이즈 제거, 전자공학회논문에서는 3-45썼지만 여러 논문에서 0.1-100Hz로 사용하길래 바꿈)
    
    Args:
        signal: 입력 ECG 신호
        fs: 샘플링 주파수
        lowcut: 하한 주파수
        highcut: 상한 주파수
        filter_order: 필터 차수
    
    Returns:
        필터링된 신호
    """
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    fir_coeff = firwin(filter_order + 1, [low, high], pass_zero=False)
    filtered_signal = filtfilt(fir_coeff, 1.0, signal)
    
    return filtered_signal


def normalize_beat(beat_signal):
    """
    개별 비트를 Z-score 정규화
    평균 0, 표준편차 1로 변환
    
    Args:
        beat_signal: ECG 비트 신호
    
    Returns:
        정규화된 신호
    """
    mean = np.mean(beat_signal)
    std = np.std(beat_signal)
    return (beat_signal - mean) / (std + 1e-8)

def preprocess_leads(sig_data, sig_names, leads, fs):
    """
    선택된 리드에 대해 밴드패스 필터 적용
    
    Args:
        sig_data: 전체 신호 데이터
        sig_names: 신호 이름 리스트
        leads: 사용할 리드 리스트
        fs: 샘플링 주파수
    
    Returns:
        dict: 리드별 필터링된 신호
    """
    filtered_data = {}
    for lead in leads:
        if lead in sig_names:
            idx = sig_names.index(lead)
            raw_signal = sig_data[:, idx]
            filtered = apply_bandpass_filter(raw_signal, fs=fs, lowcut=0.1, highcut=100.0)
            filtered_data[lead] = filtered
    return filtered_data


def extract_single_beat(filtered_data, leads, start, end, fs, fs_out, target_length):
    """
    단일 비트를 추출하고 전처리
    
    Args:
        filtered_data: 필터링된 신호 딕셔너리
        leads: 사용할 리드 리스트
        start: 시작 인덱스
        end: 종료 인덱스
        fs: 입력 샘플링 주파수
        fs_out: 출력 샘플링 주파수
        target_length: 목표 길이
    
    Returns:
        segment: 전처리된 비트 (성공 시)
        None: 유효하지 않은 경우
    """
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
    """
    MIT-BIH 레코드에서 비트 추출 및 전처리
    
    Args:
        record_path: 레코드 파일 경로
        record_name: 레코드 이름 (예: '101')
        leads: 사용할 리드 리스트 (예: ['MLII'])
        fs_out: 출력 샘플링 주파수
        segment_seconds: 세그먼트 길이 (초)
    
    Returns:
        beats: 비트 리스트
        labels: 원본 레이블 리스트
        groups: 그룹 레이블 리스트 (AAMI 표준)
    """
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
    """
    여러 레코드에서 비트 추출
    
    Args:
        data_path: 데이터 경로
        record_names: 레코드 이름 리스트
        leads: 사용할 리드
        fs_out: 출력 샘플링 주파수
        segment_seconds: 세그먼트 길이
        desc: 진행바 설명
    
    Returns:
        beats, labels, groups 리스트
    """
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


def save_dataset(output_dir, split_name, beats, labels_grouped, labels_original):
    """
    추출된 데이터를 numpy 파일로 저장
    
    Args:
        output_dir: 출력 디렉토리
        split_name: 'train' 또는 'test'
        beats: 비트 리스트
        labels_grouped: 그룹 레이블
        labels_original: 원본 레이블
    """
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    if not beats:
        return None, None, None
    
    # numpy 배열로 변환
    X = np.array(beats, dtype=np.float32)
    if X.ndim == 4 and X.shape[1] == 1:
        X = np.squeeze(X, axis=1)
    y_grouped = np.array(labels_grouped)
    y_original = np.array(labels_original)
    
    # 저장
    np.save(os.path.join(split_dir, f'{split_name}_data.npy'), X)
    np.save(os.path.join(split_dir, f'{split_name}_labels.npy'), y_grouped)
    np.save(os.path.join(split_dir, f'{split_name}_labels_original.npy'), y_original)
    
    return X, y_grouped, y_original


def process_dataset_train_test_split(data_path, output_dir, leads=['MLII'], 
                                     fs_out=360, segment_seconds=1.0):
    """
    MIT-BIH 데이터셋을 DS1 (train) / DS2 (test)로 분할하여 처리
    Inter-patient split을 위한 표준 분할 방식
    
    Args:
        data_path: MIT-BIH 데이터베이스 경로
        output_dir: 출력 디렉토리
        leads: 사용할 리드 리스트
        fs_out: 출력 샘플링 주파수
        segment_seconds: 세그먼트 길이 (초)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing MIT-BIH dataset with DS1/DS2 split...")
    print(f"DS1 Training: {len(DS1_TRAIN_RECORDS)} records")
    print(f"DS2 Testing: {len(DS2_TEST_RECORDS)} records")

    # DS1 학습 데이터 처리
    print("\nProcessing DS1 Training data...")
    train_beats, train_labels, train_groups = process_records(
        data_path, DS1_TRAIN_RECORDS, leads, fs_out, segment_seconds,
        desc="Extracting train data"
    )

    # DS2 테스트 데이터 처리
    print("\nProcessing DS2 Testing data...")
    test_beats, test_labels, test_groups = process_records(
        data_path, DS2_TEST_RECORDS, leads, fs_out, segment_seconds,
        desc="Extracting test data"
    )

    # 데이터 저장
    X_train, y_train, y_train_orig = save_dataset(
        output_dir, 'train', train_beats, train_groups, train_labels
    )
    X_test, y_test, y_test_orig = save_dataset(
        output_dir, 'test', test_beats, test_groups, test_labels
    )

    # 메타데이터 생성
    metadata = {
        'dataset_type': 'ds1_train_ds2_test_split',
        'ds1_train_records': DS1_TRAIN_RECORDS,
        'ds2_test_records': DS2_TEST_RECORDS,
        'leads': leads,
        'fs_output': fs_out,
        'segment_seconds': segment_seconds,
        'train_samples': len(X_train) if X_train is not None else 0,
        'test_samples': len(X_test) if X_test is not None else 0,
        'train_class_distribution': dict(Counter(y_train)) if y_train is not None else {},
        'test_class_distribution': dict(Counter(y_test)) if y_test is not None else {},
        'label_mapping': LABEL_GROUP_MAP
    }

    with open(os.path.join(output_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # 결과 출력
    print(f"\n=== Dataset Processing Complete ===")
    print(f"Output directory: {output_dir}")
    print(f"\nDataset Summary:")
    print(f"  Training (DS1): {len(DS1_TRAIN_RECORDS)} records, "
          f"{len(X_train) if X_train is not None else 0} samples")
    print(f"    Class distribution: {dict(Counter(y_train)) if y_train is not None else {}}")
    
    print(f"  Testing (DS2): {len(DS2_TEST_RECORDS)} records, "
          f"{len(X_test) if X_test is not None else 0} samples")
    print(f"    Class distribution: {dict(Counter(y_test)) if y_test is not None else {}}")

    print(f"\nSaved files:")
    print(f"  - train/train_data.npy, train_labels.npy, train_labels_original.npy")
    print(f"  - test/test_data.npy, test_labels.npy, test_labels_original.npy")
    print(f"  - dataset_metadata.json")

def main():
    """
    메인 실행 함수
    명령행 인자를 파싱하여 데이터셋 전처리 실행 <----- 왜여기서 하냐면 이거 다른 폴더에 있던거 병합해서 따로 되어있음
    """
    parser = argparse.ArgumentParser(
        description='MIT-BIH ECG 데이터셋 전처리 (DS1/DS2 분할)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data/mit-bih-arrhythmia-database-1.0.0',
        help='MIT-BIH 데이터베이스 경로'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./data/processed/Exp_A1_1sec',
        help='전처리된 데이터 출력 디렉토리'
    )
    parser.add_argument(
        '--leads',
        nargs='+',
        default=['MLII'],
        help='사용할 ECG 리드 (예: MLII, V1, V2)'
    )
    parser.add_argument(
        '--fs_out',
        type=int,
        default=360,
        help='출력 샘플링 주파수 (Hz)'
    )
    parser.add_argument(
        '--segment_seconds',
        type=float,
        default=1.0,
        help='세그먼트 길이 (초), 만약에 1초라고 하면 R-peak 기준 전후 0.5초씩 추출 180 + 180 = 360 샘플'
    )
    
    args = parser.parse_args()

    process_dataset_train_test_split(
        data_path=args.data_path,
        output_dir=args.output_dir,
        leads=args.leads,
        fs_out=args.fs_out,
        segment_seconds=args.segment_seconds
    )


if __name__ == "__main__":
    main()