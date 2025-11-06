"""
실험 로그 수집 및 Neptune 업로드 스크립트

기능:
1. logs 디렉토리의 모든 실험 결과 수집
2. config.json과 results 파일을 읽어 통합 CSV 생성
3. Neptune에 실험 결과 및 시각화 업로드
4. 하이퍼파라미터별 성능 분석 그래프 생성
"""

import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Neptune 관련 import
try:
    import neptune
    NEPTUNE_AVAILABLE = True
except ImportError:
    NEPTUNE_AVAILABLE = False
    print("Warning: Neptune not installed. Install with: pip install neptune")

# ==================== 설정 ====================
BASE_LOG_DIR = './logs'
OUTPUT_DIR = './log_analysis'
CONFIG_FNAME = 'config.json'
OVERALL_RESULT_FNAME = 'results_overall.csv'
PERCLASS_RESULT_FNAME = 'results_per_class.csv'
SUMMARY_FNAME = 'summary.txt'

# Neptune 설정
NEPTUNE_PROJECT = "jinjufj63/ECG-Arrhythmia"
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI2NDQ2M2Y0Ny0xNjBjLTQ1OGUtOWY3NS03ZWZjYjdmODU5NzUifQ=="

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ==================== 데이터 로딩 ====================
def find_experiment_dirs(base_dir):
    """config.json이 있는 모든 실험 디렉토리 찾기"""
    exp_dirs = []
    for root, dirs, files in os.walk(base_dir):
        if CONFIG_FNAME in files:
            exp_dirs.append(root)
    return sorted(exp_dirs)


def load_experiment_data(exp_dir):
    """실험 데이터 로드 (config + results)"""
    exp_name = os.path.basename(exp_dir)
    data = {'exp_name': exp_name, 'exp_path': exp_dir}
    
    # Config 로드
    config_path = os.path.join(exp_dir, CONFIG_FNAME)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Training 설정
        data['batch_size'] = config.get('Training', {}).get('batch_size')
        data['epochs'] = config.get('Training', {}).get('epochs')
        data['lr_initial'] = config.get('Training', {}).get('lr_initial')
        data['decay_epoch'] = config.get('Training', {}).get('decay_epoch')
        
        # Data 설정
        data['segment_seconds'] = config.get('Data', {}).get('segment_seconds')
        data['sampling_rate'] = config.get('Data', {}).get('sampling_rate')
        data['ecg_leads'] = ','.join(config.get('Data', {}).get('ecg_leads', []))
        
        # Loss 설정
        data['lambda_combined'] = config.get('Loss', {}).get('lambda_combined')
        data['loss_mode'] = config.get('Loss', {}).get('mode')
        data['proxy_type'] = config.get('Loss', {}).get('proxy_type')
        data['proxy_alpha'] = config.get('Loss', {}).get('proxy_alpha')
        data['proxy_delta'] = config.get('Loss', {}).get('proxy_delta')
    
    # Overall results 로드
    overall_path = os.path.join(exp_dir, OVERALL_RESULT_FNAME)
    if os.path.exists(overall_path):
        df_overall = pd.read_csv(overall_path)
        
        # Macro metrics
        macro_row = df_overall[df_overall['Metric_Type'] == 'Macro']
        if not macro_row.empty:
            data['macro_accuracy'] = macro_row['Accuracy'].values[0]
            data['macro_f1'] = macro_row['F1'].values[0]
            data['macro_sensitivity'] = macro_row['Sensitivity'].values[0]
            data['macro_precision'] = macro_row['Precision'].values[0]
            data['macro_specificity'] = macro_row['Specificity'].values[0]
        
        # Weighted metrics
        weighted_row = df_overall[df_overall['Metric_Type'] == 'Weighted']
        if not weighted_row.empty:
            data['weighted_accuracy'] = weighted_row['Accuracy'].values[0]
            data['weighted_f1'] = weighted_row['F1'].values[0]
            data['weighted_sensitivity'] = weighted_row['Sensitivity'].values[0]
            data['weighted_precision'] = weighted_row['Precision'].values[0]
            data['weighted_specificity'] = weighted_row['Specificity'].values[0]
    
    # Per-class results 로드
    perclass_path = os.path.join(exp_dir, PERCLASS_RESULT_FNAME)
    if os.path.exists(perclass_path):
        df_perclass = pd.read_csv(perclass_path)
        for _, row in df_perclass.iterrows():
            cls = row['Class']
            data[f'f1_{cls}'] = row['F1-Score']
            data[f'sens_{cls}'] = row['Sensitivity']
            data[f'prec_{cls}'] = row['Precision']
            # Per-class accuracy 추가
            if 'Accuracy' in row:
                data[f'acc_{cls}'] = row['Accuracy']
    
    # Best epoch 추출
    summary_path = os.path.join(exp_dir, SUMMARY_FNAME)
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            for line in f:
                if 'Best Epoch:' in line:
                    data['best_epoch'] = int(line.split(':')[1].strip())
                    break
    
    return data


# ==================== 시각화 ====================
def plot_overall_comparison(df, output_dir):
    """전체 실험 성능 비교"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Experiment Performance Comparison', fontsize=16, fontweight='bold')
    
    metrics = [
        ('macro_accuracy', 'Macro Accuracy'),
        ('macro_f1', 'Macro F1-Score'),
        ('weighted_accuracy', 'Weighted Accuracy'),
        ('weighted_f1', 'Weighted F1-Score')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        if metric in df.columns:
            data = df.sort_values(metric, ascending=False).head(20)
            bars = ax.barh(range(len(data)), data[metric])
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels(data['exp_name'], fontsize=8)
            ax.set_xlabel(title, fontsize=10)
            ax.set_title(f'{title} (Top 20)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='x')
            bars[0].set_color('red')
            for i, v in enumerate(data[metric]):
                ax.text(v, i, f' {v:.4f}', va='center', fontsize=8)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'overall_comparison.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def plot_class_performance(df, output_dir):
    """클래스별 평균 성능"""
    classes = ['N', 'S', 'V', 'F', 'Q']
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Average Performance by Class', fontsize=16, fontweight='bold')
    
    # F1-Score
    f1_scores = [df[f'f1_{cls}'].mean() if f'f1_{cls}' in df.columns else 0 for cls in classes]
    axes[0].bar(classes, f1_scores, color=sns.color_palette("husl", len(classes)))
    axes[0].set_ylabel('F1-Score', fontsize=12)
    axes[0].set_title('Average F1-Score', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(f1_scores):
        axes[0].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Sensitivity
    sens_scores = [df[f'sens_{cls}'].mean() if f'sens_{cls}' in df.columns else 0 for cls in classes]
    axes[1].bar(classes, sens_scores, color=sns.color_palette("husl", len(classes)))
    axes[1].set_ylabel('Sensitivity', fontsize=12)
    axes[1].set_title('Average Sensitivity', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(sens_scores):
        axes[1].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Per-class Accuracy (새로 추가)
    acc_scores = [df[f'acc_{cls}'].mean() if f'acc_{cls}' in df.columns else 0 for cls in classes]
    axes[2].bar(classes, acc_scores, color=sns.color_palette("husl", len(classes)))
    axes[2].set_ylabel('Accuracy', fontsize=12)
    axes[2].set_title('Average Per-class Accuracy', fontsize=13)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(acc_scores):
        axes[2].text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'class_performance.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


def plot_hyperparameter_effects(df, output_dir):
    """하이퍼파라미터별 성능 변화"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Hyperparameter Effects on Performance', fontsize=16, fontweight='bold')
    
    params = [
        ('segment_seconds', 'Segment Length (s)'),
        ('batch_size', 'Batch Size'),
        ('lr_initial', 'Learning Rate'),
        ('epochs', 'Epochs'),
        ('lambda_combined', 'Lambda Combined'),
        ('loss_mode', 'Loss Mode')
    ]
    
    for idx, (param, title) in enumerate(params):
        ax = axes[idx // 3, idx % 3]
        
        if param in df.columns and 'macro_f1' in df.columns:
            data = df[[param, 'macro_f1']].dropna()
            
            if len(data) > 0:
                if param == 'loss_mode':
                    # 카테고리형 변수
                    grouped = data.groupby(param)['macro_f1'].mean().sort_values(ascending=False)
                    ax.bar(range(len(grouped)), grouped.values)
                    ax.set_xticks(range(len(grouped)))
                    ax.set_xticklabels(grouped.index, rotation=45, ha='right', fontsize=9)
                else:
                    # 수치형 변수
                    if data[param].nunique() <= 10:
                        grouped = data.groupby(param)['macro_f1'].mean().sort_index()
                        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, markersize=8)
                    else:
                        ax.scatter(data[param], data['macro_f1'], alpha=0.6, s=50)
                
                ax.set_xlabel(title, fontsize=10)
                ax.set_ylabel('Macro F1-Score', fontsize=10)
                ax.set_title(title, fontsize=11)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'hyperparameter_effects.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    return filepath


# ==================== Neptune 업로드 ====================
def upload_to_neptune(df, project_name, api_token, plot_paths):
    """실험 결과를 Neptune에 업로드"""
    if not NEPTUNE_AVAILABLE:
        print("Neptune is not available. Skipping upload.")
        return
    
    if not api_token:
        print("NEPTUNE_API_TOKEN not found. Skipping upload.")
        return
    
    print(f"\nUploading {len(df)} experiments to Neptune...")
    
    for idx, row in df.iterrows():
        try:
            # Neptune run 생성
            run = neptune.init_run(
                project=project_name,
                api_token=api_token,
                name=row['exp_name'],
                tags=["batch_upload", "archived"]
            )
            
            # Config 정보
            run["parameters/batch_size"] = row.get('batch_size')
            run["parameters/epochs"] = row.get('epochs')
            run["parameters/lr_initial"] = row.get('lr_initial')
            run["parameters/segment_seconds"] = row.get('segment_seconds')
            run["parameters/loss_mode"] = row.get('loss_mode')
            run["parameters/proxy_type"] = row.get('proxy_type')
            run["parameters/sampling_rate"] = row.get('sampling_rate')
            
            # Overall 성능 지표
            if pd.notna(row.get('macro_accuracy')):
                run["metrics/macro/accuracy"] = float(row['macro_accuracy'])
            if pd.notna(row.get('macro_f1')):
                run["metrics/macro/f1"] = float(row['macro_f1'])
            if pd.notna(row.get('macro_sensitivity')):
                run["metrics/macro/sensitivity"] = float(row['macro_sensitivity'])
            if pd.notna(row.get('macro_precision')):
                run["metrics/macro/precision"] = float(row['macro_precision'])
            if pd.notna(row.get('macro_specificity')):
                run["metrics/macro/specificity"] = float(row['macro_specificity'])
            
            if pd.notna(row.get('weighted_accuracy')):
                run["metrics/weighted/accuracy"] = float(row['weighted_accuracy'])
            if pd.notna(row.get('weighted_f1')):
                run["metrics/weighted/f1"] = float(row['weighted_f1'])
            if pd.notna(row.get('weighted_sensitivity')):
                run["metrics/weighted/sensitivity"] = float(row['weighted_sensitivity'])
            if pd.notna(row.get('weighted_precision')):
                run["metrics/weighted/precision"] = float(row['weighted_precision'])
            if pd.notna(row.get('weighted_specificity')):
                run["metrics/weighted/specificity"] = float(row['weighted_specificity'])
            
            # Per-class 성능
            classes = ['N', 'S', 'V', 'F', 'Q']
            for cls in classes:
                if f'f1_{cls}' in row and pd.notna(row[f'f1_{cls}']):
                    run[f"metrics/class_{cls}/f1"] = float(row[f'f1_{cls}'])
                if f'sens_{cls}' in row and pd.notna(row[f'sens_{cls}']):
                    run[f"metrics/class_{cls}/sensitivity"] = float(row[f'sens_{cls}'])
                if f'prec_{cls}' in row and pd.notna(row[f'prec_{cls}']):
                    run[f"metrics/class_{cls}/precision"] = float(row[f'prec_{cls}'])
            
            # Best epoch
            if 'best_epoch' in row and pd.notna(row['best_epoch']):
                run["training/best_epoch"] = int(row['best_epoch'])
            
            run.stop()
            print(f"  [{idx+1}/{len(df)}] Uploaded: {row['exp_name']}")
            
        except Exception as e:
            print(f"  Error uploading {row['exp_name']}: {e}")
    
    # 전체 분석 그래프를 별도 run으로 업로드
    print("\nUploading analysis plots to Neptune...")
    try:
        run = neptune.init_run(
            project=project_name,
            api_token=api_token,
            name=f"Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tags=["analysis", "summary"]
        )
        
        # 전체 통계
        run["summary/total_experiments"] = len(df)
        if 'macro_f1' in df.columns:
            run["summary/best_macro_f1"] = float(df['macro_f1'].max())
            run["summary/avg_macro_f1"] = float(df['macro_f1'].mean())
        if 'macro_accuracy' in df.columns:
            run["summary/best_accuracy"] = float(df['macro_accuracy'].max())
            run["summary/avg_accuracy"] = float(df['macro_accuracy'].mean())
        
        # 플롯 업로드
        for plot_name, plot_path in plot_paths.items():
            if os.path.exists(plot_path):
                run[f"plots/{plot_name}"].upload(plot_path)
        
        run.stop()
        print("Analysis plots uploaded successfully")
        
    except Exception as e:
        print(f"Error uploading analysis plots: {e}")


# ==================== 메인 ====================
def main():
    print("="*70)
    print("Experiment Log Analysis with Neptune")
    print("="*70)
    
    # 1. 실험 디렉토리 찾기
    print(f"\n[1/5] Searching experiments in {BASE_LOG_DIR}...")
    exp_dirs = find_experiment_dirs(BASE_LOG_DIR)
    print(f"Found {len(exp_dirs)} experiments")
    
    # 2. 데이터 로드
    print(f"\n[2/5] Loading experiment data...")
    all_data = [load_experiment_data(exp_dir) for exp_dir in exp_dirs]
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} experiments")
    
    # 3. CSV 저장
    print(f"\n[3/5] Saving summary CSV...")
    csv_path = os.path.join(OUTPUT_DIR, 'all_experiments_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # 4. 시각화 생성
    print(f"\n[4/5] Generating visualizations...")
    plot_paths = {}
    plot_paths['overall_comparison'] = plot_overall_comparison(df, OUTPUT_DIR)
    plot_paths['class_performance'] = plot_class_performance(df, OUTPUT_DIR)
    plot_paths['hyperparameter_effects'] = plot_hyperparameter_effects(df, OUTPUT_DIR)
    print(f"Saved {len(plot_paths)} plots to {OUTPUT_DIR}/")
    
    # 5. Neptune 업로드
    print(f"\n[5/5] Uploading to Neptune...")
    if NEPTUNE_AVAILABLE and NEPTUNE_API_TOKEN:
        upload_to_neptune(df, NEPTUNE_PROJECT, NEPTUNE_API_TOKEN, plot_paths)
        print("Neptune upload completed")
    else:
        print("Neptune upload skipped")
    
    # 통계 출력
    print("\n" + "="*70)
    print("Summary Statistics:")
    print(f"  Total experiments: {len(df)}")
    if 'macro_f1' in df.columns:
        best_idx = df['macro_f1'].idxmax()
        print(f"  Best Macro F1: {df.loc[best_idx, 'macro_f1']:.4f} ({df.loc[best_idx, 'exp_name']})")
        print(f"  Avg Macro F1: {df['macro_f1'].mean():.4f}")
    if 'macro_accuracy' in df.columns:
        print(f"  Best Accuracy: {df['macro_accuracy'].max():.4f}")
        print(f"  Avg Accuracy: {df['macro_accuracy'].mean():.4f}")
    print("="*70)


if __name__ == '__main__':
    main()