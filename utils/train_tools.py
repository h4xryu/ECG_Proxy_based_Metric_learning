"""
학습 관련 유틸리티 함수 모음
손실 계산, 체크포인트 관리, 결과 출력 등
"""
import os
from datetime import datetime
import torch
from torch import nn
from loss_functions import *
from proxy_inference import proxy_test


class TrainTools:
    """학습 관련 정적 유틸리티 메서드 모음"""
    
    @staticmethod
    def setup_losses(opt):
        """
        옵션 설정에 따라 손실 함수 초기화
        
        Args:
            opt: 옵션 객체
            
        Returns:
            cross_entropy_loss: Cross Entropy 손실 함수
            proxy_loss: Proxy 손실 함수 (없으면 None)
        """
        cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean')
        proxy_loss = None
        
        if opt.proxy_weight > 0 or opt.loss_type != 'CE':
            if opt.proxy_type == 'ProxyAnchorLoss':
                proxy_loss = ProxyAnchorLoss(
                    alpha=opt.proxy_alpha,
                    delta=opt.proxy_delta
                )
            elif opt.proxy_type == 'FocalStyleProxyAnchorLoss':
                proxy_loss = FocalStyleProxyAnchorLoss(
                    alpha=opt.proxy_alpha,
                    delta=opt.proxy_delta,
                    pos_gamma=opt.proxy_pos_gamma,
                    neg_gamma=opt.proxy_neg_gamma
                )
            elif opt.proxy_type == 'MultiProxyAnchorLoss':
                proxy_loss = MultiProxyAnchorLoss(
                    num_classes=opt.classes,
                    num_proxies_per_class=opt.multi_proxy_num_proxies_per_class,
                    embed_dim=opt.out_ch,
                    alpha=opt.proxy_alpha,
                    delta=opt.proxy_delta,
                    topk_pos=opt.multi_proxy_topk_pos,
                    topk_neg=opt.multi_proxy_topk_neg,
                    softplus_threshold=opt.multi_proxy_softplus_threshold
                )
            else:
                raise ValueError(f"Unknown proxy type: {opt.proxy_type}")
        
        return cross_entropy_loss, proxy_loss
    
    @staticmethod
    def compute_loss(outputs, features, labels, opt, cross_entropy_loss, 
                     proxy_loss, model_proxies, device):
        """
        옵션 설정에 따라 손실 계산
        
        Args:
            outputs: 모델 출력 [B, nOUT]
            features: 특징 벡터 [B, out_ch]
            labels: 레이블 [B]
            opt: 옵션 객체
            cross_entropy_loss: CE 손실 함수
            proxy_loss: Proxy 손실 함수
            model_proxies: 모델의 proxy 파라미터
            device: 디바이스
            
        Returns:
            total_loss: 전체 손실
            proxy_loss_val: Proxy 손실 값
            ce_loss: CE 손실 값
        """
        ce_loss = torch.tensor(0.0, device=device)
        proxy_loss_val = torch.tensor(0.0, device=device)
        
        # Loss type에 따른 계산
        if opt.loss_type == 'CE':
            # Cross Entropy만 사용
            ce_loss = opt.cross_entropy_weight * cross_entropy_loss(outputs, labels)
            total_loss = ce_loss
            
        elif opt.loss_type == 'proxy':
            # Proxy loss만 사용
            if proxy_loss is not None:
                if opt.proxy_type == 'MultiProxyAnchorLoss':
                    proxy_loss_val = opt.proxy_weight * proxy_loss(features, labels)
                else:
                    proxy_loss_val = opt.proxy_weight * proxy_loss(
                        features, labels, model_proxies
                    )
            total_loss = proxy_loss_val
            
        elif opt.loss_type == 'combined' or opt.proxy_combined:
            # CE + Proxy 결합
            ce_loss = opt.cross_entropy_weight * cross_entropy_loss(outputs, labels)
            
            if proxy_loss is not None:
                if opt.proxy_type == 'MultiProxyAnchorLoss':
                    proxy_loss_val = opt.proxy_weight * proxy_loss(features, labels)
                else:
                    proxy_loss_val = opt.proxy_weight * proxy_loss(
                        features, labels, model_proxies
                    )
            
            total_loss = ce_loss + proxy_loss_val
        else:
            raise ValueError(f"Unknown loss type: {opt.loss_type}")
        
        return total_loss, proxy_loss_val, ce_loss
    
    @staticmethod
    def get_predictions(outputs, features, model_proxies, opt):
        """
        모델 출력으로부터 예측 레이블 추출
        
        Args:
            outputs: 모델 출력
            features: 특징 벡터
            model_proxies: 모델 proxy 파라미터
            opt: 옵션 객체
            
        Returns:
            predictions: 예측 레이블
        """
        if opt.proxy_combined:
            # Proxy inference 사용
            predictions = proxy_test(features, model_proxies)
        else:
            # 일반 argmax 사용
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions
    
    @staticmethod
    def print_metrics(title, class_names, metrics, avg_loss, proxy_loss, ce_loss):
        """
        메트릭 출력
        
        Args:
            title: 출력할 제목
            class_names: 클래스 이름 리스트
            metrics: 메트릭 딕셔너리
            avg_loss: 평균 손실
            proxy_loss: Proxy 손실
            ce_loss: CE 손실
        """
        print(f"\n{title}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Proxy Loss: {proxy_loss:.4f}")
        print(f"  CE Loss: {ce_loss:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"  Weighted Sensitivity: {metrics['weighted_sensitivity']:.4f}")
        print(f"  Weighted Specificity: {metrics['weighted_specificity']:.4f}")
        
        print("\n  Per-class Sensitivity:")
        for name, sens in zip(class_names, metrics['sensitivity']):
            print(f"    {name}: {sens:.4f}")
        
        print("\n  Per-class Specificity:")
        for name, spec in zip(class_names, metrics['specificity']):
            print(f"    {name}: {spec:.4f}")
    
    @staticmethod
    def print_test_results(all_class_metrics, nsv_class_metrics, class_names,
                          avg_loss, proxy_loss, ce_loss):
        """
        테스트 결과 출력
        
        Args:
            all_class_metrics: 전체 클래스 메트릭
            nsv_class_metrics: N, S, V 클래스만의 메트릭
            class_names: 클래스 이름 리스트
            avg_loss: 평균 손실
            proxy_loss: Proxy 손실
            ce_loss: CE 손실
        """
        print("\n" + "="*70)
        print("FINAL TEST RESULTS")
        print("="*70)
        
        TrainTools.print_metrics(
            "ALL 5 CLASSES (N, S, V, F, Q)", 
            class_names, 
            all_class_metrics, 
            avg_loss, 
            proxy_loss, 
            ce_loss
        )
        TrainTools.print_metrics(
            "N, S, V 3 CLASSES ONLY", 
            ['N', 'S', 'V'], 
            nsv_class_metrics, 
            avg_loss, 
            proxy_loss, 
            ce_loss
        )
        
        print("="*70)
    
    @staticmethod
    def create_log_directory():
        """
        타임스탬프 기반 로그 디렉토리 생성
        
        Returns:
            tensorboard 로그 디렉토리 경로
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(os.getcwd(), 'logs', f'{timestamp}')
        
        os.makedirs(os.path.join(log_dir, 'tensorboard'), exist_ok=True)
        os.makedirs(os.path.join(log_dir, 'checkpoints'), exist_ok=True)
        
        return os.path.join(log_dir, 'tensorboard')
    
    @staticmethod
    def get_model_dir(log_dir):
        """
        모델 저장 디렉토리 경로 가져오기
        
        Args:
            log_dir: 로그 디렉토리 경로
            
        Returns:
            체크포인트 저장 디렉토리 경로
        """
        return os.path.join(
            os.path.dirname(log_dir.replace('tensorboard', 'checkpoints')), 
            'checkpoints'
        )
    
    @staticmethod
    def create_checkpoint(model, optimizer, scheduler, epoch, f1_score, 
                         best_f1, class_names):
        """
        체크포인트 딕셔너리 생성
        
        Args:
            model: 모델
            optimizer: 옵티마이저
            scheduler: 스케줄러
            epoch: 현재 에포크
            f1_score: 현재 F1 스코어
            best_f1: 최고 F1 스코어
            class_names: 클래스 이름 리스트
            
        Returns:
            체크포인트 딕셔너리
        """
        return {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'f1_score': f1_score,
            'best_f1': best_f1,
            'class_names': class_names
        }
    
    @staticmethod
    def save_checkpoint(checkpoint, model_dir, epoch, is_best):
        """
        체크포인트 저장
        
        Args:
            checkpoint: 체크포인트 딕셔너리
            model_dir: 모델 저장 디렉토리
            epoch: 현재 에포크
            is_best: 최고 성능 모델 여부
            
        Returns:
            best_model_path: 최고 모델 경로 (is_best=True인 경우), 아니면 None
        """
        # 에포크별 체크포인트 저장
        torch.save(checkpoint, os.path.join(model_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        # 최고 모델 저장
        if is_best:
            best_model_path = os.path.join(model_dir, 'best_model.pth')
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with F1: {checkpoint['f1_score']:.4f}")
            return best_model_path
        
        return None
    
    @staticmethod
    def load_best_model(model, best_model_path, device):
        """
        최고 성능 모델 체크포인트 로드
        
        Args:
            model: 모델 (가중치를 로드할)
            best_model_path: 최고 모델 경로
            device: 디바이스
        """
        if best_model_path is None:
            print("No best model found. Using current model for testing.")
            return
        
        print(f"Loading best model from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    @staticmethod
    def print_training_config(opt):
        """
        학습 설정 출력
        
        Args:
            opt: 옵션 객체
        """
        print("\n" + "="*70)
        print("TRAINING CONFIGURATION")
        print("="*70)
        print(f"Loss Type: {opt.loss_type}")
        print(f"Proxy Type: {opt.proxy_type if hasattr(opt, 'proxy_type') else 'N/A'}")
        print(f"Proxy Weight: {opt.proxy_weight}")
        print(f"CE Weight: {opt.cross_entropy_weight}")
        print(f"Batch Size: {opt.batch_size}")
        print(f"Learning Rate: {opt.lr_initial}")
        print(f"Epochs: {opt.nepoch}")
        print(f"Device: {opt.device}")
        print("="*70 + "\n")

