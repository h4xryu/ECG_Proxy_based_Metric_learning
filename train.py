import argparse
import os
import random
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix, classification_report
from models import *
from utils import *
from loss_functions import *
from opt import Options
from proxy_inference import *
from utils.metrics import MetricsCalculator
from utils.data_utils import ECGDataManager
from utils.train_tools import TrainTools
os.environ["QT_QPA_PLATFORM"] = "offscreen"

"""
================================================================================
TRAINER CLASS - 학습 클래스
================================================================================
"""


class Trainer:
    """
    ECG 분류 모델 학습 및 평가 클래스
    """
    
    def __init__(self, opt):
        """
        Args:
            opt: 학습 옵션 객체
        """
        self.opt = opt
        self.device = opt.device
        self.class_names = ECGDataManager.get_class_names(opt)
        
        self._setup_model()
        self._setup_data()
        self._setup_losses()
        self._setup_optimizer()
        self._setup_logger()
        self._init_tracking_vars()

    """================================================================================학습세팅================================================================================"""

    def _setup_model(self):
        """모델 초기화"""
        use_proxy = self.opt.proxy_weight > 0 or self.opt.loss_type != 'CE'
        self.model = UNet(
            nOUT=self.opt.classes,
            in_channels=self.opt.inputs,
            out_ch=self.opt.out_ch,
            mid_ch=self.opt.mid_ch,
            inconv_size=self.opt.inconv_size,
            rub0_layers=self.opt.r0_layer,
            use_proxy=use_proxy
        ).to(self.device)
    
    def _setup_data(self):
        """데이터 로더 초기화"""
        self.train_loader, self.valid_loader = ECGDataManager.load_train_valid_data(
            self.opt.path_train_data,
            self.opt.path_train_labels,
            self.opt.batch_size
        )

        self.test_loader = ECGDataManager.load_test_data(
            self.opt.path_test_data,
            self.opt.path_test_labels,
            self.opt.batch_size
        )
    
    
    def _setup_optimizer(self):
        """옵티마이저 및 스케줄러 초기화"""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.opt.lr_initial,
            weight_decay=0.04,
            betas=(0.9, 0.999)
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=self.opt.decay_epoch,
            gamma=0.1
        )
    
    def _setup_logger(self):
        """로거 초기화"""
        log_dir = TrainTools.create_log_directory()
        self.logger = TensorBoardLogger(log_dir, self.opt, self.class_names)
    
    def _init_tracking_vars(self):
        """추적 변수 초기화"""
        self.best_valid_f1 = 0.0
        self.best_metrics = {}
        self.best_model_path = None

    def train(self):
        """
        전체 학습 루프 실행
        
        에포크별로 학습, 검증을 수행하고 최고 성능 모델을 저장합니다.
        학습 완료 후 테스트를 자동으로 수행합니다.
        """
        TrainTools.print_training_config(self.opt)

        for epoch in range(1, self.opt.nepoch + 1):
            start_time = time.time()

            # 에포크별 학습 및 검증
            train_loss = self._train_epoch(epoch)
            valid_metrics, valid_loss = self._evaluate()
            
            # 최고 성능 모델 추적
            is_best = valid_metrics['macro_f1'] > self.best_valid_f1
            if is_best:
                self.best_valid_f1 = valid_metrics['macro_f1']
                self.best_metrics = valid_metrics.copy()
                self.best_metrics['epoch'] = epoch

            # 체크포인트 저장
            self._save_checkpoint(epoch, valid_metrics['macro_f1'], is_best)

            # 학습률 스케줄링
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
            
            # 로깅
            epoch_time = time.time() - start_time
            train_metrics = {'loss': train_loss}
            valid_metrics['loss'] = valid_loss
            self.logger.log_epoch(epoch, train_metrics, valid_metrics, current_lr, epoch_time)

        # 학습 완료 후 최고 모델로 테스트
        print('\nTraining completed. Starting test evaluation with best model...')
        self._test_with_best_model()

        # 실험 정리
        self.logger.finalize_experiment(self.best_metrics)
        self.logger.close()
        
        print('Training and testing completed.')
        print(f"Best validation F1: {self.best_valid_f1:.4f} at epoch {self.best_metrics.get('epoch', 'N/A')}")

    def _compute_loss(self, outputs, features, labels):

        return TrainTools.compute_loss(
            outputs=outputs,
            features=features,
            labels=labels,
            opt=self.opt,
            cross_entropy_loss=self.cross_entropy_loss,
            proxy_loss=self.proxy_loss,
            model_proxies=self.model.get_proxies() if self.opt.proxy_weight > 0 else None,
            device=self.device
        )
    
    def _train_epoch(self, epoch):
        """
        한 에포크 학습
        
        Args:
            epoch: 현재 에포크 번호
            
        Returns:
            평균 학습 손실
        """
        self.model.train()
        total_loss = 0
        
        for X, Y in TrainingBar(self.train_loader):
            X, Y = X.float().to(self.device), Y.long().to(self.device)
            
            outputs, features = self.model(X)
            loss, proxy_loss, ce_loss = self._compute_loss(outputs, features, Y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.logger.log_loss(loss.item(), proxy_loss.item(), ce_loss.item(), epoch)
        
        return total_loss / len(self.train_loader)

    def _evaluate(self, data_loader=None):
        """
        모델 평가 (검증 또는 테스트)
        
        Args:
            data_loader: 평가할 데이터 로더 (None이면 valid_loader 사용)
            
        Returns:
            metrics: 평가 메트릭 딕셔너리
            avg_loss: 평균 손실
        """
        if data_loader is None:
            data_loader = self.valid_loader
        
        self.model.eval()
        pred_labels, true_labels, pred_probas = [], [], []
        total_loss = 0
        
        with torch.no_grad():
            for X, Y in Bar(desc="Evaluation", dataloader=data_loader):
                X, Y = X.float().to(self.device), Y.long().to(self.device)
                
                outputs, features = self.model(X)
                loss, _, _ = self._compute_loss(outputs, features, Y)
                total_loss += loss.item()

                # 확률 계산
                probas = torch.softmax(outputs, dim=1)
                
                # 예측 레이블 추출
                preds = TrainTools.get_predictions(
                    outputs, features, 
                    self.model.get_proxies() if self.opt.proxy_weight > 0 else None,
                    self.opt
                )
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(Y.cpu().numpy())
                pred_probas.extend(probas.cpu().numpy())
        
        pred_labels = np.array(pred_labels)
        true_labels = np.array(true_labels)
        pred_probas = np.array(pred_probas)
        avg_loss = total_loss / len(data_loader)
        
        metrics = self.logger.calculate_metrics(true_labels, pred_labels, pred_probas)
        
        return metrics, avg_loss

    def _load_best_model(self):
        """최고 성능 모델 체크포인트 로드"""
        TrainTools.load_best_model(self.model, self.best_model_path, self.device)
    
    def _predict(self, data_loader):
        """
        데이터 로더로부터 예측 수행
        
        Args:
            data_loader: 예측할 데이터 로더
            
        Returns:
            pred_labels: 예측 레이블
            true_labels: 실제 레이블
            avg_loss: 평균 손실
            proxy_loss: Proxy 손실
            ce_loss: CE 손실
        """
        self.model.eval()
        pred_labels, true_labels = [], []
        total_loss = 0
        proxy_loss_sum = 0
        ce_loss_sum = 0
        
        with torch.no_grad():
            for X, Y in Bar(desc="Testing", dataloader=data_loader):
                X, Y = X.float().to(self.device), Y.long().to(self.device)
                
                outputs, features = self.model(X)
                loss, proxy_loss, ce_loss = self._compute_loss(outputs, features, Y)
                total_loss += loss.item()
                proxy_loss_sum += proxy_loss.item()
                ce_loss_sum += ce_loss.item()
           
                # 예측 레이블 추출
                preds = TrainTools.get_predictions(
                    outputs, features,
                    self.model.get_proxies() if self.opt.proxy_weight > 0 else None,
                    self.opt
                )
                pred_labels.extend(preds.cpu().numpy())
                true_labels.extend(Y.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader)
        avg_proxy_loss = proxy_loss_sum / len(data_loader)
        avg_ce_loss = ce_loss_sum / len(data_loader)
        
        return (np.array(pred_labels), np.array(true_labels), 
                avg_loss, avg_proxy_loss, avg_ce_loss)
    
    def _test_with_best_model(self):
        """
        최고 성능 모델로 테스트 수행
        """
        self._load_best_model()
        pred_labels, true_labels, avg_loss, proxy_loss, ce_loss  = self._predict(self.test_loader)
        
        all_class_metrics = MetricsCalculator.calculate_weighted_metrics(
            true_labels, pred_labels
        )
        nsv_class_metrics = MetricsCalculator.calculate_weighted_metrics(
            true_labels, pred_labels, class_indices=[0, 1, 2]   #N, S, V 클래스만의 메트릭
        )
        
        TrainTools.print_test_results(
            all_class_metrics, nsv_class_metrics, 
            self.class_names, avg_loss, proxy_loss, ce_loss
        )
        
        if hasattr(self.logger, 'log_test_results'):
            test_metrics = {
                'all_classes': all_class_metrics,
                'nsv_classes': nsv_class_metrics
            }
            self.logger.log_test_results(test_metrics, avg_loss, proxy_loss, ce_loss * self.opt.cross_entropy_weight)
    

    def _save_checkpoint(self, epoch, f1_score, is_best):
        """
        체크포인트 저장
        
        Args:
            epoch: 현재 에포크
            f1_score: 현재 F1 스코어
            is_best: 최고 성능 모델 여부
        """
        model_dir = TrainTools.get_model_dir(self.logger.log_dir)
        checkpoint = TrainTools.create_checkpoint(
            self.model, self.optimizer, self.scheduler,
            epoch, f1_score, self.best_valid_f1, self.class_names
        )
        
        best_path = TrainTools.save_checkpoint(checkpoint, model_dir, epoch, is_best)
        if best_path:
            self.best_model_path = best_path

    def _setup_losses(self):
        """손실 함수 초기화"""
        self.cross_entropy_loss, self.proxy_loss = TrainTools.setup_losses(self.opt)

"""
================================================================================
메인 실행 함수
================================================================================
"""

def set_seed(seed=1234):
    """
    재현성을 위한 시드 고정
    
    Args:
        seed: 랜덤 시드 값 (기본값: 1234)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    """
    명령행 인자 파싱
    
    Returns:
        opt: 파싱된 옵션 객체
    """
    parser = argparse.ArgumentParser(description='ECG Classification Training')
    options = Options().init(parser)
    return options.parse_known_args()[0]


def main():
    """
    메인 실행 함수
    ECG 분류 모델 학습 파이프라인 실행
    """
    set_seed()
    opt = parse_arguments()
    
    trainer = Trainer(opt)
    trainer.train()


if __name__ == '__main__':
    main()