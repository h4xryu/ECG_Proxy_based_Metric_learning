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
from scipy.signal import firwin, filtfilt, medfilt
import pywt
from torch.utils.data import DataLoader
from dataloader import ECGDataloader
import matplotlib.pyplot as plt


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




def preprocess_leads(sig_data, sig_names, leads, fs):
    """선택된 리드에 대해 DWT 필터 적용 (논문 방식)"""
    filtered_data = {}
    for lead in leads:
        if lead in sig_names:
            idx = sig_names.index(lead)
            raw_signal = sig_data[:, idx]
            
            # DWT with db6 웨이블릿으로 baseline wander와 노이즈 제거
            # signal_filtered = remove_baseline_bandpassfilter(raw_signal, fs=360, lowcut=0.1, highcut=100.0, filter_order=256)
            signal_filtered = remove_baseline_median(raw_signal, fs = 360, win_ms1 = 200, win_ms2 = 600)
            # signal_filtered = remove_baseline_dwt(raw_signal, wavelet='db6', level=9)
            
            filtered_data[lead] = signal_filtered
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
    
    # Single lead: (target_length,) 형태 - normalize 제거 (논문에 없음)
    if len(leads) == 1:
        segment = segment[0].astype(np.float32)
    # Multi lead: (num_leads, target_length) 형태
    else:
        segment = np.stack(segment).astype(np.float32)
    
    return segment





def extract_5s_segments_old(record_path, record_name, leads, fs_out=360, segment_seconds=5.0):
    """MIT-BIH 레코드에서 R-peak 중심 세그먼트 추출 및 전처리
    
    R-peak Aligned 방식:
    - 각 R-peak annotation을 중심으로 세그먼트 추출
    - 예: 1800 샘플이면 R-peak 기준 -900, +900 샘플
    - DWT with db6로 전처리
    """
    # 어노테이션 및 신호 데이터 로드
    ann_data = wfdb.rdann(os.path.join(record_path, record_name), 'atr')
    sig_data, meta = wfdb.rdsamp(os.path.join(record_path, record_name))
    fs = meta['fs']
    sig_names = meta['sig_name']
    
    # 리드별 필터링 (DWT with db6)
    filtered_data = preprocess_leads(sig_data, sig_names, leads, fs)
    
    # 파라미터 설정
    target_length = int(segment_seconds * fs_out)  # 5초 * 360Hz = 1800 샘플
    half_segment = int(segment_seconds * fs / 2)  # R-peak 앞뒤로 절반씩
    
    segments, labels, groups = [], [], []
    total_samples = len(sig_data)
    
    # 각 R-peak annotation을 중심으로 세그먼트 추출
    for r_peak_pos, symbol in zip(ann_data.sample, ann_data.symbol):
 
        # 유효한 심볼만 처리
        if symbol not in LABEL_GROUP_MAP:
            continue
        
        segment_group = LABEL_GROUP_MAP[symbol]
        
        # R-peak를 중심으로 앞뒤로 균등하게 자르기
        start = r_peak_pos - half_segment
        end = r_peak_pos + half_segment
        
        # 경계 체크
        if start < 0 or end > total_samples:
            continue
        
        # 세그먼트 추출
        segment_data = []
        print(leads)
        for lead in leads:
            if lead not in filtered_data:
                segment_data = None
                break
            
            lead_signal = filtered_data[lead][start:end]
            
            # 길이 확인
            if len(lead_signal) != (end - start):
                segment_data = None
                break
            
            # 리샘플링
            lead_signal_resampled = resample_signal(lead_signal, fs, fs_out, target_length)
            plt.plot(lead_signal_resampled)
    
            plt.plot(r_peak_pos)
            plt.show()
            segment_data.append(lead_signal_resampled)
        
        if segment_data is None:
            continue
        
        # Single lead: (target_length,) 형태 - normalize 제거 (논문에 없음)
        if len(leads) == 1:
            segment_array = segment_data[0].astype(np.float32)
        # Multi lead: (num_leads, target_length) 형태
        else:
            segment_array = np.stack(segment_data).astype(np.float32)
        
        segments.append(segment_array)
        labels.append(symbol)
        groups.append(segment_group)
    
    return segments, labels, groups


"""
================================================================================
1-1. Baseline wander 제거
================================================================================
"""


def remove_baseline_bandpassfilter(signal, fs=360, lowcut=0.1, highcut=100.0, filter_order=256):
    nyquist = fs / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    fir_coeff = firwin(filter_order + 1, [low, high], pass_zero=False)
    filtered_signal = filtfilt(fir_coeff, 1.0, signal)
    
    return filtered_signal


def remove_baseline_median(x: np.ndarray, fs: int, win_ms1: int = 200, win_ms2: int = 600) -> np.ndarray:
    """
    Remove baseline using cascaded median filters.
    
    Args:
        x: Input array, shape (L,) or (N, L)
        fs: Sampling frequency in Hz
        win_ms1: First median filter window in ms
        win_ms2: Second median filter window in ms
    
    Returns:
        Baseline-corrected signal with same shape as input
    """
    # 1D 입력을 2D로 변환
    input_1d = False
    if x.ndim == 1:
        input_1d = True
        x = x[np.newaxis, :]  # (L,) -> (1, L)
    
    assert x.ndim == 2, "x must be 1D or 2D array"
    N, L = x.shape
    
    def _ms_to_odd_k(ms: int) -> int:
        k = int(round(ms * fs / 1000.0))
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        # 커널이 신호길이를 넘지 않도록 보정
        if k > L:
            k = L if L % 2 == 1 else L - 1
            if k < 3:
                k = 3
        return k

    k1 = _ms_to_odd_k(win_ms1)  # ~200ms
    k2 = _ms_to_odd_k(win_ms2)  # ~600ms

    # 각 샘플(행)별로 1D median 필터 두 번 적용
    y = np.empty_like(x, dtype=np.float32)
    for i in range(N):
        s = x[i].astype(np.float32, copy=False)
        m1 = medfilt(s, kernel_size=k1)
        baseline = medfilt(m1, kernel_size=k2)
        y[i] = (s - baseline)

    # 입력이 1D였으면 1D로 반환
    if input_1d:
        y = y.squeeze(0)  # (1, L) -> (L,)
    
    return y.copy()

def remove_baseline_dwt(signal, wavelet='db6', level=9):
    """DWT (Discrete Wavelet Transform)를 사용한 denoising
    
    논문: Daubechies 6 (db6) 웨이블릿을 사용하여 baseline wander와 노이즈 제거
    """
    # DWT 분해
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # 임계값 기반 denoising (soft thresholding)
    # 첫 번째 계수(근사 계수)는 유지, 나머지 상세 계수들만 thresholding
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    
    # DWT 재구성
    denoised_signal = pywt.waverec(coeffs, wavelet)
    
    # 원본 신호와 길이를 맞춤
    return denoised_signal[:len(signal)]



"""
================================================================================
1-2. Segment 추출
================================================================================
"""


def extract_5s_segments(record_path, record_name, leads, fs_out=360, segment_seconds=5.0, plot_debug=False):
    """MIT-BIH 레코드에서 R-peak 중심 세그먼트 추출 및 전처리 (라벨링 시각화 포함)
    
    R-peak Aligned 방식:
    - 각 R-peak annotation을 중심으로 세그먼트 추출
    - 예: 1800 샘플이면 R-peak 기준 -900, +900 샘플
    - DWT with db6로 전처리
    
    키보드 컨트롤 (plot_debug=True 시):
    - w: 이전 세그먼트
    - e: 다음 세그먼트
    - r: 현재 플롯 저장
    - p: 10개 스킵
    - o: 100개 스킵
    - q: 종료
    """
    # 어노테이션 및 신호 데이터 로드
    ann_data = wfdb.rdann(os.path.join(record_path, record_name), 'atr')
    sig_data, meta = wfdb.rdsamp(os.path.join(record_path, record_name))
    fs = meta['fs']
    sig_names = meta['sig_name']

    
    # 리드별 필터링 (DWT with db6)
    filtered_data = preprocess_leads(sig_data, sig_names, leads, fs)
    
    # 파라미터 설정
    target_length = int(segment_seconds * fs_out)  # 5초 * 360Hz = 1800 샘플
    half_segment = int(segment_seconds * fs / 2)  # R-peak 앞뒤로 절반씩
    
    segments, labels, groups, label_list = [], [], [], []
    plot_data_list = []  # plot_debug를 위한 데이터 저장
    total_samples = len(sig_data)
    
    # 각 R-peak annotation을 중심으로 세그먼트 추출
    for r_peak_pos, symbol in zip(ann_data.sample, ann_data.symbol):
 
        # 유효한 심볼만 처리
        if symbol not in LABEL_GROUP_MAP:
            continue
        
        segment_group = LABEL_GROUP_MAP[symbol]
        
        # R-peak를 중심으로 앞뒤로 균등하게 자르기
        start = r_peak_pos - half_segment
        end = r_peak_pos + half_segment
        
        # 경계 체크
        if start < 0 or end > total_samples:
            continue
        
        # 세그먼트 추출
        segment_data = []
        for lead in leads:
            if lead not in filtered_data:
                segment_data = None
                break
            
            lead_signal = filtered_data[lead][start:end]
            
            # 길이 확인
            if len(lead_signal) != (end - start):
                segment_data = None
                break
            
            # 리샘플링
            lead_signal_resampled = resample_signal(lead_signal, fs, fs_out, target_length)
            segment_data.append(lead_signal_resampled)
        
        if segment_data is None:
            continue
        
        # Single lead: (target_length,) 형태
        if len(leads) == 1:
            segment_array = segment_data[0].astype(np.float32)
        # Multi lead: (num_leads, target_length) 형태
        else:
            segment_array = np.stack(segment_data).astype(np.float32)
        
        # 라벨링 시각화 - 해당 세그먼트 내의 모든 비트 표시
        beats_in_segment = []
        for peak_pos, peak_symbol in zip(ann_data.sample, ann_data.symbol):
            if start <= peak_pos < end and peak_symbol in LABEL_GROUP_MAP:
                # 세그먼트 내 상대 위치 계산 (리샘플링된 좌표로 변환)
                relative_pos = (peak_pos - start) / (end - start) * target_length
                beats_in_segment.append((relative_pos, peak_symbol, LABEL_GROUP_MAP[peak_symbol]))
        
        # plot_debug를 위한 데이터 저장
        if plot_debug and beats_in_segment:
            plot_data_list.append({
                'segment': segment_array.copy(),
                'beats': beats_in_segment.copy(),
                'leads': leads
            })
        
        segments.append(segment_array)
        labels.append(symbol)
        groups.append(segment_group)
    
    # plot_debug 모드에서 인터랙티브 시각화
    if plot_debug and plot_data_list:
        plt.ion()  # Interactive mode ON
        fig, ax = plt.subplots(figsize=(12, 6))
        
        class SegmentViewer:
            def __init__(self):
                self.idx = 0
                self.quit_flag = False
                self.update_plot()
            
            def update_plot(self):
                if self.quit_flag or not (0 <= self.idx < len(plot_data_list)):
                    return
                
                ax.clear()
                
                data = plot_data_list[self.idx]
                segment_array = data['segment']
                beats_in_segment = data['beats']
                
                # 신호 플롯
                if len(data['leads']) == 1:
                    ax.plot(segment_array, linewidth=1)
                else:
                    ax.plot(segment_array[0], linewidth=1)
                
                # 비트 마커 표시
                y_max = ax.get_ylim()[1]
                for beat_pos, beat_symbol, beat_group in beats_in_segment:
                    ax.axvline(x=beat_pos, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                    ax.text(beat_pos, y_max, f'{beat_symbol}({beat_group})', 
                           rotation=0, verticalalignment='top', fontsize=10)
                
                ax.set_title(f'Segment {self.idx+1}/{len(plot_data_list)} | w:prev e:next r:save p:+10 o:+100 q:quit', 
                           fontsize=12, fontweight='bold')
                ax.set_xlabel('Sample')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)
                
                plt.draw()
                plt.pause(0.001)  # UI 업데이트를 위한 짧은 pause
            
            def on_key(self, event):
                if event.key == 'w':  # 이전
                    self.idx = max(0, self.idx - 1)
                    self.update_plot()
                elif event.key == 'e':  # 다음
                    self.idx = min(len(plot_data_list) - 1, self.idx + 1)
                    self.update_plot()
                elif event.key == 'r':  # 저장
                    save_path = f'segment_{record_name}_{self.idx}.png'
                    plt.savefig(save_path, dpi=150, bbox_inches='tight')
                    print(f'Saved: {save_path}')
                elif event.key == 'p':  # 10개 스킵
                    self.idx = min(len(plot_data_list) - 1, self.idx + 10)
                    self.update_plot()
                elif event.key == 'o':  # 100개 스킵
                    self.idx = min(len(plot_data_list) - 1, self.idx + 100)
                    self.update_plot()
                elif event.key == 'q':  # 종료
                    self.quit_flag = True
                    plt.close('all')
                    print("Viewer closed.")
        
        viewer = SegmentViewer()
        fig.canvas.mpl_connect('key_press_event', viewer.on_key)
        
        print("\n=== Interactive Viewer Started ===")
        print("Controls: w(prev) | e(next) | r(save) | p(+10) | o(+100) | q(quit)")
        print("Close the plot window or press 'q' to exit.\n")
        
        plt.show(block=True)  # 사용자가 종료할 때까지 대기
        plt.ioff()  # Interactive mode OFF
    
    return segments, labels, groups


# 이전 방식도 유지 (호환성)
def extract_beats(record_path, record_name, leads, fs_out=360, segment_seconds=1.0, plot_debug=False):
    """MIT-BIH 레코드에서 비트 추출 및 전처리
    
    논문 방식으로 5초 세그먼트를 사용하려면 extract_5s_segments 사용
    """
    # 5초 세그먼트 방식 사용 (논문 방식)
    if segment_seconds >= 3.0:
        return extract_5s_segments(record_path, record_name, leads, fs_out, segment_seconds, plot_debug)
    
    # 기존 방식 (단일 비트 추출) - 하위 호환성
    ann_data = wfdb.rdann(os.path.join(record_path, record_name), 'atr')
    sig_data, meta = wfdb.rdsamp(os.path.join(record_path, record_name))
    fs = meta['fs']
    sig_names = meta['sig_name']
    
    filtered_data = preprocess_leads(sig_data, sig_names, leads, fs)
    
    target_length = int(segment_seconds * fs_out)
    samples_before = int(round(90 * fs / 360.0))
    samples_after = int(round(110 * fs / 360.0))
    
    beats, labels, groups = [], [], []
    
    for pos, symbol in zip(ann_data.sample, ann_data.symbol):
        if symbol not in LABEL_GROUP_MAP:
            continue
        
        start = pos - samples_before
        end = pos + samples_after
        
        if start < 0 or end > len(sig_data):
            continue
        
        segment = extract_single_beat(
            filtered_data, leads, start, end, fs, fs_out, target_length
        )
        
        if segment is not None:
            beats.append(segment)
            labels.append(symbol)
            groups.append(LABEL_GROUP_MAP[symbol])
    
    return beats, labels, groups


def process_records(data_path, record_names, leads, fs_out, segment_seconds, desc="Processing", plot_debug=False):
    """여러 레코드에서 비트 추출"""
    all_beats, all_labels, all_groups = [], [], []
    
    for record_name in tqdm(record_names, desc=desc):
        try:
            beats, labels, groups = extract_beats(
                data_path, record_name, leads, fs_out, segment_seconds, plot_debug
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
    MIT-BIH 데이터를 전처리 (캐시 없이 매번 새로 전처리)
    """
    cache_dir = get_cache_dir(output_base_dir, segment_seconds, fs_out, leads)
    
    # 파일 경로 정의
    train_data_path = os.path.join(cache_dir, 'train', 'train_data.npy')
    train_labels_path = os.path.join(cache_dir, 'train', 'train_labels.npy')
    test_data_path = os.path.join(cache_dir, 'test', 'test_data.npy')
    test_labels_path = os.path.join(cache_dir, 'test', 'test_labels.npy')
    
    # 기존 캐시가 있으면 삭제
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
    
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
    매번 새로 전처리
    """
    from train import (path_mitbih_raw, path_processed_base, 
                       segment_seconds, fs_out, ecg_leads)
    
    # 전처리 (매번 새로)
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
    train_data에서 이미 전처리된 데이터 사용
    """
    from train import (path_mitbih_raw, path_processed_base, 
                       segment_seconds, fs_out, ecg_leads)
    
    # 전처리된 데이터 경로 가져오기
    cache_dir = get_cache_dir(path_processed_base, segment_seconds, fs_out, ecg_leads)
    
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


def plot_sample_ecg(sample_idx=0, dataset='train', save_path='sample_ecg.png'):
    """
    전처리된 ECG 샘플 시각화
    
    Args:
        sample_idx: 샘플 인덱스
        dataset: 'train' 또는 'test'
        save_path: 저장 경로
    """
    from train import (path_mitbih_raw, path_processed_base, 
                       segment_seconds, fs_out, ecg_leads)
    
    # 전처리된 데이터 경로
    cache_dir = get_cache_dir(path_processed_base, segment_seconds, fs_out, ecg_leads)
    
    # 데이터 로드
    data_path = os.path.join(cache_dir, dataset, f'{dataset}_data.npy')
    labels_path = os.path.join(cache_dir, dataset, f'{dataset}_labels.npy')
    
    data = np.load(data_path)
    labels = np.load(labels_path)
    
    # 샘플 선택
    sample = data[sample_idx]
    label = labels[sample_idx]
    
    # Plot
    plt.figure(figsize=(15, 5))
    
    # 시간축 생성 (초 단위)
    time_axis = np.linspace(0, segment_seconds, len(sample))
    
    plt.plot(time_axis, sample, linewidth=1.5, color='blue')
    plt.title(f'ECG Sample #{sample_idx} - Label: {label} - 5s Segment (1800 samples @ 360Hz)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 통계 정보 표시
    stats_text = f'Mean: {sample.mean():.4f}\nStd: {sample.std():.4f}\nMin: {sample.min():.4f}\nMax: {sample.max():.4f}'
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
             fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ ECG 샘플 저장됨: {save_path}")
    print(f"  - 데이터셋: {dataset}")
    print(f"  - 샘플 인덱스: {sample_idx}")
    print(f"  - 레이블: {label}")
    print(f"  - Shape: {sample.shape}")
    print(f"  - Duration: {segment_seconds}s")
    print(f"  - Sampling Rate: {fs_out}Hz")
    
    plt.close()
    
    return sample, label


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
    """학습 설정 출력 - 50 에포크 학습용"""
    from train import (proxy_type, proxy_alpha, proxy_delta,
                       lambda_combined, batch_size, lr_initial, nepoch, device,
                       segment_seconds, fs_out, ecg_leads)
    

    # 데이터 설정
    print(f"\n┌─ DATA PREPROCESSING " + "─"*67 + "┐")
    print(f"│  Segment Length   : {segment_seconds}s ({int(segment_seconds * fs_out)} samples at {fs_out}Hz)")
    print(f"│  Sampling Rate    : {fs_out} Hz")
    print(f"│  ECG Leads        : {', '.join(ecg_leads)}")
    print("└" + "─"*88 + "┘")
    
    # 손실 함수 설정
    print(f"\n┌─ LOSS FUNCTION " + "─"*72 + "┐")
    print(f"│  λ (lambda)       : {lambda_combined:.2f}")
    
    if lambda_combined == 1.0:
        print(f"│  Mode             : CE only (Cross-Entropy)")
    elif lambda_combined == 0.0:
        print(f"│  Mode             : Proxy only")
        print(f"│  Proxy Type       : {proxy_type}")
        print(f"│  Proxy α (alpha)  : {proxy_alpha}")
        print(f"│  Proxy δ (delta)  : {proxy_delta}")
    else:
        print(f"│  Mode             : Combined")
        print(f"│    → CE weight    : {lambda_combined:.2f}")
        print(f"│    → Proxy weight : {1-lambda_combined:.2f}")
        print(f"│  Proxy Type       : {proxy_type}")
        print(f"│  Proxy α (alpha)  : {proxy_alpha}")
        print(f"│  Proxy δ (delta)  : {proxy_delta}")
    print("└" + "─"*88 + "┘")
    
    # 학습 설정
    print(f"\n┌─ TRAINING PARAMETERS " + "─"*66 + "┐")
    print(f"│  Batch Size       : {batch_size}")
    print(f"│  Learning Rate    : {lr_initial}")
    print(f"│  Total Epochs     : {nepoch}")
    print(f"│  Device           : {device}")
    print("└" + "─"*88 + "┘")
    
    print("="*70 + "\n")



if __name__ == '__main__':
    DS1_TRAIN_RECORDS = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119', '122', '124',
    '201', '203', '205', '207', '208', '209', '215', '220', '223', '230'
    ]
    
    
    train_beats, train_labels, train_groups = process_records(
        './data/mit-bih-arrhythmia-database-1.0.0', DS1_TRAIN_RECORDS, ['MLII'], 360, 5,
        desc="DS1 Train",
        plot_debug=True
    )

 
