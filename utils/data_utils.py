import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataloader import ECGDataloader
from utils.preprocessing import label2index

"""
================================================================================
ECG DATA MANAGER CLASS - ECG 데이터 관리 클래스
================================================================================
"""  

class ECGDataManager:
    """
    ECG 데이터 로딩 및 전처리를 담당하는 클래스
    데이터 분할, 로더 생성 등의 기능 제공
    """
    
    @staticmethod
    def load_train_valid_data(path_train_data, path_train_labels, batch_size,
                               test_size=0.2, random_state=1234, num_workers=5):
        """
        학습 데이터 로드 및 train/validation 분할
        
        Args:
            path_train_data: 학습 데이터 경로
            path_train_labels: 학습 레이블 경로
            batch_size: 배치 크기
            test_size: validation 비율
            random_state: 랜덤 시드
            num_workers: 데이터 로더 워커 수
        
        Returns:
            train_loader: 학습 데이터 로더
            valid_loader: 검증 데이터 로더
        """
        train_data = np.load(path_train_data, allow_pickle=True)
        train_labels = np.load(path_train_labels, allow_pickle=True)
        
        # 채널 차원 추가 및 레이블 변환
        train_data = np.expand_dims(train_data, axis=1)
        train_labels = np.array([label2index(i) for i in train_labels])
        
        # 계층 샘플링으로 train/validation 분할
        X_train, X_val, y_train, y_val = train_test_split(
            train_data, train_labels,
            test_size=test_size,
            random_state=random_state,
            stratify=train_labels
        )
        
        print(f"Train set size: {len(X_train)}, Validation set size: {len(X_val)}")
        
        # 데이터로더 생성
        train_loader = DataLoader(
            ECGDataloader(X_train, y_train),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        valid_loader = DataLoader(
            ECGDataloader(X_val, y_val),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, valid_loader
    
    @staticmethod
    def load_test_data(path_test_data, path_test_labels, batch_size, num_workers=0):
        """
        테스트 데이터 로드
        
        Args:
            path_test_data: 테스트 데이터 경로
            path_test_labels: 테스트 레이블 경로
            batch_size: 배치 크기
            num_workers: 데이터 로더 워커 수
        
        Returns:
            test_loader: 테스트 데이터 로더
        """
        test_data = np.load(path_test_data)
        test_labels = np.load(path_test_labels)
        
        # 채널 차원 추가 및 레이블 변환
        test_labels = np.array([label2index(i) for i in test_labels])
        test_data = np.expand_dims(test_data, axis=1)

        test_loader = DataLoader(
            ECGDataloader(test_data, test_labels),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return test_loader
    
    @staticmethod
    def get_class_names(opt):
        """
        클래스 이름 목록 반환
        
        Args:
            opt: 옵션 객체
        
        Returns:
            list: 클래스 이름 리스트
        """
        ecg_classes = ['N', 'S', 'V', 'F', 'Q']
        
        if hasattr(opt, 'classes'):
            if opt.classes <= len(ecg_classes):
                return ecg_classes[:opt.classes]
            else:
                return ecg_classes + [f'Class_{i}' for i in range(len(ecg_classes), opt.classes)]
        
        return [f'Class_{i}' for i in range(5)]

