# import argparse
# import os
# import numpy as np
# import torch
# from torch.utils.data import DataLoader
# from sklearn.metrics import confusion_matrix, classification_report
# from sklearn.manifold import TSNE
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import seaborn as sns
# from models import *
# from utils import *
# from loss_functions import *
# from opt import Options
# from dataloader import ECGDataloader
# from proxy_inference import *
# from utils import *

# class ECGTester:
#     def __init__(self, opt, model_path):
#         self.opt = opt
#         self.device = opt.device
#         self.model_path = model_path
#         self.class_names = self._get_class_names()
        
#         # 모델 초기화
#         self.model = ResUNet(
#             nOUT=opt.classes,
#             in_channels=opt.inputs,
#             out_ch=opt.out_ch,
#             mid_ch=opt.mid_ch,
#             inconv_size=opt.inconv_size,
#             rub0_layers=opt.r0_layer  
#         ).to(opt.device)  
#         # self.model = HUnivModel(nOUT=opt.classes, in_channels=opt.inputs).to(opt.device)
#         # 모델 로드
#         self._load_model()
        
#         # 테스트 데이터 로더
#         self.test_loader = self._load_test_data()

#     def _get_class_names(self):
#         ecg_classes = ['N', 'S', 'V', 'F', 'Q']
        
#         if hasattr(self.opt, 'classes'):
#             if self.opt.classes <= len(ecg_classes):
#                 return ecg_classes[:self.opt.classes]
#             else:
#                 return ecg_classes + [f'Class_{i}' for i in range(len(ecg_classes), self.opt.classes)]
        
#         return [f'Class_{i}' for i in range(5)]

#     def _load_model(self):
#         """지정된 경로에서 모델 로드"""
#         print(f"Loading model from {self.model_path}")
#         checkpoint = torch.load(self.model_path, map_location=self.device)
        
#         # 체크포인트 구조 확인
#         if 'model_state_dict' in checkpoint:
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             print("Loaded model_state_dict from checkpoint")
            
#             # 클래스 이름이 저장되어 있다면 로드
#             if 'class_names' in checkpoint:
#                 self.class_names = checkpoint['class_names']
#                 print(f"Loaded class names: {self.class_names}")
#         else:
#             # 직접 state_dict가 저장된 경우
#             self.model.load_state_dict(checkpoint)
#             print("Loaded direct state_dict")

#     def _load_test_data(self):
#         """테스트 데이터 로더 생성"""
#         test_data = np.load(self.opt.path_test_data)
#         test_labels = np.load(self.opt.path_test_labels)
#         test_labels = np.array([label2index(i) for i in test_labels])
#         test_data = np.expand_dims(test_data, axis=1)

#         test_loader = DataLoader(
#             ECGDataloader(test_data, test_labels),
#             batch_size=self.opt.batch_size,
#             shuffle=False,
#             num_workers=0
#         )
        
#         return test_loader

#     def _calculate_sensitivity_specificity(self, y_true, y_pred, num_classes):
#         cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
#         sensitivity, specificity = [], []
#         for i in range(num_classes):
#             tp = cm[i, i]
#             fn = cm[i, :].sum() - tp
#             fp = cm[:, i].sum() - tp
#             tn = cm.sum() - tp - fn - fp
#             sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#             sensitivity.append(sens)
#             specificity.append(spec)
#         return sensitivity, specificity
    
#     def calculate_sensitivity_specificity(self, y_true, y_pred, num_classes):
#         """클래스별 sensitivity와 specificity 계산"""
#         cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
        
#         sensitivity = []
#         specificity = []
        
#         for i in range(num_classes):
#             tp = cm[i, i]
#             fn = cm[i, :].sum() - tp
#             fp = cm[:, i].sum() - tp
#             tn = cm.sum() - tp - fn - fp
            
#             sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#             sensitivity.append(sens)
            
#             spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
#             specificity.append(spec)
        
#         return sensitivity, specificity

#     def extract_features_before_fc(self):
#         """FC 레이어 전 feature 추출"""
#         print("Extracting features before FC layer...")
        
#         self.model.eval()
#         all_features = []
#         all_labels = []
        
#         with torch.no_grad():
#             for X, Y in Bar(desc="Extracting features", dataloader=self.test_loader):
#                 X = X.float().to(self.device)
#                 Y = Y.long().to(self.device)
                
#                 # # Forward pass until just before FC layer
#                 x = X.squeeze(0) if X.dim() == 4 else X
#                 x = F.leaky_relu(self.model.bn(self.model.conv(x)))
#                 x = self.model.aspp(x)
#                 x = self.model.rub_0(x)
#                 x = F.dropout(x, p=0.5, training=False)
#                 # Pooling to get final features before FC
#                 features = self.model.maxpool(x).squeeze(2)  # [batch_size, feature_dim]

#                 # x = self.model.conv7(self.model.conv6(self.model.conv5(self.model.conv4(self.model.conv3(self.model.conv2(self.model.conv1(x)))))))
    
#                 # features = self.model.pool(x)
#                 # features = features.squeeze(2)
                
#                 all_features.append(features.cpu().numpy())
#                 all_labels.extend(Y.cpu().numpy())
        
#         all_features = np.vstack(all_features)
#         all_labels = np.array(all_labels)
        
#         return all_features, all_labels

#     def visualize_tsne(self, features, labels, save_dir):
#         """t-SNE 시각화"""
#         print("Performing t-SNE visualization...")
        
#         # t-SNE 적용
#         tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
#         tsne_features = tsne.fit_transform(features)
        
#         # 시각화
#         plt.figure(figsize=(12, 10))
        
#         # 클래스별로 다른 색상과 마커 사용
#         colors = plt.cm.Set3(np.linspace(0, 1, len(self.class_names)))
#         markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
        
#         for i, class_name in enumerate(self.class_names):
#             mask = labels == i
#             if np.any(mask):
#                 plt.scatter(
#                     tsne_features[mask, 0], 
#                     tsne_features[mask, 1],
#                     c=[colors[i]], 
#                     marker=markers[i % len(markers)],
#                     label=f'{class_name} (n={np.sum(mask)})',
#                     alpha=0.7,
#                     s=50
#                 )
        
#         plt.title('t-SNE Visualization of Features Before FC Layer', fontsize=16)
#         plt.xlabel('t-SNE Component 1', fontsize=12)
#         plt.ylabel('t-SNE Component 2', fontsize=12)
#         plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         # 저장
#         tsne_path = os.path.join(save_dir, 'tsne_features_before_fc.png')
#         plt.savefig(tsne_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"t-SNE plot saved: {tsne_path}")
        
#         # 추가: 클래스별 분리도 분석
#         # self._analyze_class_separation(tsne_features, labels, save_dir)

#     def _analyze_class_separation(self, tsne_features, labels, save_dir):
#         """클래스별 분리도 분석"""
#         print("Analyzing class separation...")
        
#         # 클래스별 중심점 계산
#         class_centers = []
#         for i in range(len(self.class_names)):
#             mask = labels == i
#             if np.any(mask):
#                 center = np.mean(tsne_features[mask], axis=0)
#                 class_centers.append(center)
#             else:
#                 class_centers.append(np.array([0, 0]))
        
#         class_centers = np.array(class_centers)
        
#         # 클래스 간 거리 행렬 계산
#         n_classes = len(self.class_names)
#         distance_matrix = np.zeros((n_classes, n_classes))
        
#         for i in range(n_classes):
#             for j in range(n_classes):
#                 if i != j:
#                     distance_matrix[i, j] = np.linalg.norm(class_centers[i] - class_centers[j])
        
#         # 히트맵으로 시각화
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(
#             distance_matrix, 
#             annot=True, 
#             fmt='.2f',
#             xticklabels=self.class_names,
#             yticklabels=self.class_names,
#             cmap='viridis'
#         )
#         plt.title('Class Separation Distance Matrix (t-SNE space)', fontsize=14)
#         plt.tight_layout()
        
#         separation_path = os.path.join(save_dir, 'class_separation_matrix.png')
#         plt.savefig(separation_path, dpi=300, bbox_inches='tight')
#         plt.close()
        
#         print(f"Class separation analysis saved: {separation_path}")

#     def test_and_visualize(self):
#         """테스트 수행"""
#         print("Starting test evaluation...")
        
#         # 결과 저장 디렉토리 생성
#         base_dir = './test_results_visualization'
#         os.makedirs(base_dir, exist_ok=True)
        
#         # 맞춘 것과 틀린 것 디렉토리 생성
#         correct_dir = os.path.join(base_dir, 'correct_predictions')
#         incorrect_dir = os.path.join(base_dir, 'incorrect_predictions')
#         os.makedirs(correct_dir, exist_ok=True)
#         os.makedirs(incorrect_dir, exist_ok=True)
        
#         # # 1. Feature 추출 및 t-SNE 시각화
#         # features, feature_labels = self.extract_features_before_fc()
#         # self.visualize_tsne(features, feature_labels, base_dir)
        
#         # 테스트 수행
#         self.model.eval()
#         pred_labels, true_labels = [], []
        
#         correct_count = 0  # 맞춘 예측의 개수 추적
#         incorrect_count = 0  # 틀린 예측의 개수 추적
        
#         with torch.no_grad():
#             for batch_idx, (X, Y) in enumerate(Bar(desc="Testing", dataloader=self.test_loader)):
#                 X = X.float().to(self.device)
#                 Y = Y.long().to(self.device)
                
#                 outputs,features_test = self.model(X)
                
#                 probas = torch.softmax(outputs, dim=1)
#                 pred_classes = torch.argmax(probas, dim=1)
#                 # pred_classes = proxy_test(features_test, self.model.get_proxies())
                
#                 # CPU로 이동하여 저장
#                 batch_pred = pred_classes.cpu().numpy()
#                 batch_true = Y.cpu().numpy()
#                 batch_data = X.cpu().numpy()
                
#                 pred_labels.extend(batch_pred)
#                 true_labels.extend(batch_true)
                
#                 # # 배치 내 각 샘플에 대해 시각화
#                 # for i in range(len(batch_pred)):
#                 #     sample_idx = batch_idx * self.opt.batch_size + i
#                 #     true_label = batch_true[i]
#                 #     pred_label = batch_pred[i]
#                 #     ecg_data = batch_data[i, 0, :]  # [seq_len] - single channel ECG
                    
#                 #     is_correct = (true_label == pred_label)
                    
#                 #     if is_correct and correct_count < 5000:
#                 #         # 맞춘 경우 - 10개만 저장
#                 #         self._save_ecg_plot(
#                 #             ecg_data, true_label, pred_label, sample_idx,
#                 #             correct_dir, "Correct"
#                 #         )
#                 #         correct_count += 1
#                 #     elif not is_correct:
#                 #         # 틀린 경우 - 모두 저장
#                 #         self._save_ecg_plot(
#                 #             ecg_data, true_label, pred_label, sample_idx,
#                 #             incorrect_dir, "Incorrect"
#                 #         )
#                 #         incorrect_count += 1 
        
#         # 결과를 numpy array로 변환
#         pred_labels = np.array(pred_labels)
#         true_labels = np.array(true_labels)
        
#         # 성능 메트릭 계산
#         accuracy = np.mean(pred_labels == true_labels)
        
#         # Classification Report 생성 (소수점 4자리까지)
#         class_report = classification_report(
#             true_labels, 
#             pred_labels, 
#             target_names=self.class_names,
#             digits=4,
#             output_dict=False
#         )
        
#         # 결과 출력
#         print("\n" + "="*60)
#         print("TEST RESULTS")
#         print("="*60)
#         print(f"Total samples: {len(true_labels)}")
#         print(f"Test Accuracy: {accuracy:.4f}")
        
#         print("\nDetailed Classification Report:")
#         print("-" * 60)
#         print(class_report)
        
#         # 혼동 행렬 출력
#         cm = confusion_matrix(true_labels, pred_labels)
#         print("\nConfusion Matrix:")
#         print("Predicted ->")
#         print("Actual |")
#         header = "    " + " ".join([f"{name:>6}" for name in self.class_names])
#         print(header)
#         for i, class_name in enumerate(self.class_names):
#             row = f"{class_name:>2} |" + " ".join([f"{cm[i,j]:>6}" for j in range(len(self.class_names))])
#             print(row)
        
#         print("="*60)
#         self._test_with_best_model()
        

#     def _save_ecg_plot(self, ecg_data, true_label, pred_label, sample_idx, save_dir, prediction_type):
#         """ECG 플롯 저장"""
#         plt.figure(figsize=(12, 4))
#         plt.plot(ecg_data, 'b-', linewidth=1.5)
#         plt.title(f'{prediction_type} Prediction - Sample {sample_idx}\n'
#                  f'True: {self.class_names[true_label]}, Predicted: {self.class_names[pred_label]}')
#         plt.xlabel('Time Steps')
#         plt.ylabel('Amplitude')
#         plt.grid(True, alpha=0.3)
#         plt.tight_layout()
        
#         filename = f'sample_{sample_idx:06d}_true_{self.class_names[true_label]}_pred_{self.class_names[pred_label]}.png'
#         plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
#         plt.close()

#     def _calculate_weighted_metrics(self, y_true, y_pred, class_indices=None):
#         if class_indices is not None:
#             mask = np.isin(y_true, class_indices)
#             y_true_filtered = y_true[mask]
#             y_pred_filtered = y_pred[mask]
#             label_mapping = {old: new for new, old in enumerate(class_indices)}
#             y_true_final = np.array([label_mapping[y] for y in y_true_filtered])
#             y_pred_final = np.array([label_mapping[y] if y in label_mapping else -1 for y in y_pred_filtered])
#             valid_mask = y_pred_final != -1
#             y_true_final = y_true_final[valid_mask]
#             y_pred_final = y_pred_final[valid_mask]
#             num_classes = len(class_indices)
#         else:
#             # 항상 모델의 총 클래스 수로 고정
#             y_true_final = y_true
#             y_pred_final = y_pred
#             num_classes = len(self.class_names)  # or self.opt.classes

#         if len(y_true_final) == 0:
#             return {
#                 'accuracy': 0.0, 'weighted_f1': 0.0,
#                 'sensitivity': [0.0] * num_classes,
#                 'specificity': [0.0] * num_classes,
#                 'weighted_sensitivity': 0.0, 'weighted_specificity': 0.0
#             }

#         # accuracy
#         accuracy = np.mean(y_true_final == y_pred_final)

#         # per-class sens/spec (축 강제)
#         sensitivity, specificity = self._calculate_sensitivity_specificity(
#             y_true_final, y_pred_final, num_classes
#         )

#         # weighted F1 (labels 명시)
#         from sklearn.metrics import f1_score
#         weighted_f1 = f1_score(
#             y_true_final, y_pred_final,
#             labels=list(range(num_classes)),
#             average='weighted', zero_division=0
#         )

#         # weighted sens/spec
#         class_counts = np.bincount(y_true_final, minlength=num_classes)
#         total = class_counts.sum()
#         if total > 0:
#             weights = class_counts / total
#             weighted_sensitivity = float(np.average(sensitivity, weights=weights))
#             weighted_specificity = float(np.average(specificity, weights=weights))
#         else:
#             weighted_sensitivity = 0.0
#             weighted_specificity = 0.0

#         return {
#             'accuracy': accuracy,
#             'weighted_f1': weighted_f1,
#             'sensitivity': sensitivity,
#             'specificity': specificity,
#             'weighted_sensitivity': weighted_sensitivity,
#             'weighted_specificity': weighted_specificity
#         }
    
#     def _test_with_best_model(self):
#         """===================================================모델 테스트==================================================="""

#         print(f"Loading best model from {'./best_model.pth}'}")
#         checkpoint = torch.load("./best_model.pth", map_location=self.device)
#         self.model.load_state_dict(checkpoint['model_state_dict'])
        
#         # 테스트 평가 - 예측값 획득
#         self.model.eval()
#         pred_labels, true_labels = [], []
        
#         with torch.no_grad():
#             for X, Y in Bar(desc="Testing", dataloader=self.test_loader):
#                 X = X.float().to(self.device)
#                 Y = Y.long().to(self.device)
                
#                 outputs, features = self.model(X)
#                 probas = torch.softmax(outputs, dim=1)
#                 pred_classes = torch.argmax(probas, dim=1)
#                 # pred_classes = proxy_test(features, self.model.get_proxies())
                
#                 pred_labels.extend(pred_classes.cpu().numpy())
#                 true_labels.extend(Y.cpu().numpy())
        
#         pred_labels = np.array(pred_labels)
#         true_labels = np.array(true_labels)
        
#         # 전체 5개 클래스 (N, S, V, F, Q) 평가
#         all_class_metrics = self._calculate_weighted_metrics(true_labels, pred_labels)
        
#         # N, S, V 3개 클래스만 평가 (클래스 인덱스: 0, 1, 2)
#         nsv_class_metrics = self._calculate_weighted_metrics(true_labels, pred_labels, class_indices=[0, 1, 2])
        
#         # 결과 출력
#         print("\n" + "="*70)
#         print("FINAL TEST RESULTS")
#         print("="*70)
        
#         # 전체 5개 클래스 결과
#         print("\n🔹 ALL 5 CLASSES (N, S, V, F, Q):")
#         print(f"  Accuracy: {all_class_metrics['accuracy']:.4f}")
#         print(f"  Weighted F1: {all_class_metrics['weighted_f1']:.4f}")
#         print(f"  Weighted Sensitivity: {all_class_metrics['weighted_sensitivity']:.4f}")
#         print(f"  Weighted Specificity: {all_class_metrics['weighted_specificity']:.4f}")
        
#         print("\n  Per-class Sensitivity:")
#         for i, (class_name, sens) in enumerate(zip(self.class_names, all_class_metrics['sensitivity'])):
#             print(f"    {class_name}: {sens:.4f}")
        
#         print("\n  Per-class Specificity:")
#         for i, (class_name, spec) in enumerate(zip(self.class_names, all_class_metrics['specificity'])):
#             print(f"    {class_name}: {spec:.4f}")
        
#         # N, S, V 3개 클래스 결과
#         print("\n🔹 N, S, V 3 CLASSES ONLY:")
#         print(f"  Accuracy: {nsv_class_metrics['accuracy']:.4f}")
#         print(f"  Weighted F1: {nsv_class_metrics['weighted_f1']:.4f}")
#         print(f"  Weighted Sensitivity: {nsv_class_metrics['weighted_sensitivity']:.4f}")
#         print(f"  Weighted Specificity: {nsv_class_metrics['weighted_specificity']:.4f}")
        
#         nsv_class_names = ['N', 'S', 'V']
#         print("\n  Per-class Sensitivity (N, S, V):")
#         for i, (class_name, sens) in enumerate(zip(nsv_class_names, nsv_class_metrics['sensitivity'])):
#             print(f"    {class_name}: {sens:.4f}")
        
#         print("\n  Per-class Specificity (N, S, V):")
#         for i, (class_name, spec) in enumerate(zip(nsv_class_names, nsv_class_metrics['specificity'])):
#             print(f"    {class_name}: {spec:.4f}")
        
#         print("="*70)
 


# def main():
#     # 명령행 인자 파싱
#     parser = argparse.ArgumentParser(description='ECG Test Only with Classification Report')
    
#     # 모델 경로 인자 추가

#     options = Options().init(parser)
#     opt = options.parse_known_args()[0]
    
#     print("ECG Test Configuration:")
#     print(f"Model path: {opt.model_path}")
#     print(f"Test data path: {opt.path_test_data}")
#     print(f"Test labels path: {opt.path_test_labels}")
#     print(f"Device: {opt.device}")
#     print(f"Classes: {opt.classes}")
    
#     # 모델 경로 존재 확인
#     if not os.path.exists(opt.model_path):
#         print(f"Error: Model file not found at {opt.model_path}")
#         return
    
#     # 테스트 데이터 경로 존재 확인
#     if not os.path.exists(opt.path_test_data):
#         print(f"Error: Test data file not found at {opt.path_test_data}")
#         return
    
#     if not os.path.exists(opt.path_test_labels):
#         print(f"Error: Test labels file not found at {opt.path_test_labels}")
#         return
    
#     # 테스트 시작
#     tester = ECGTester(opt, opt.model_path)
#     tester.test_and_visualize()
    
#     print("Test completed successfully!")

# if __name__ == '__main__':
#     main()