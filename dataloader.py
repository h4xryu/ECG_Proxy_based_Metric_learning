import torch

"""
================================================================================
ECG DATALOADER CLASS - ECG 데이터 로더 클래스
================================================================================
"""

class ECGDataloader:
    """
    PyTorch 데이터로더용 ECG 데이터셋 클래스
    numpy 배열을 받아 torch 텐서로 변환하여 제공
    """
    
    def __init__(self, data, label):
        """
        Args:
            data: ECG 신호 데이터 (numpy array)
            label: 클래스 레이블 (numpy array)
        """
        self.data = data
        self.label = label

    def __getitem__(self, index):
        """인덱스에 해당하는 데이터와 레이블 반환"""
        return (
            torch.tensor(self.data[index], dtype=torch.float),
            torch.tensor(self.label[index], dtype=torch.float)
        )

    def __len__(self):
        """전체 데이터 개수 반환"""
        return len(self.data)