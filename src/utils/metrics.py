"""
Evaluation metrics and utilities for model comparison.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def compute_metrics(y_true, y_pred, y_proba=None, average='macro'):
    """
    Compute comprehensive metrics for binary classification.
    
    Returns:
        dict: Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['roc_auc'] = np.nan
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, labels=None, save_path=None):
    """
    Plot and optionally save confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels or ['Low Risk', 'High Risk'],
                yticklabels=labels or ['Low Risk', 'High Risk'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()
    return cm

def save_metrics(metrics_dict, model_name, output_dir='outputs'):
    """
    Save metrics to CSV file, appending if file exists.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / 'metrics.csv'
    
    # Create DataFrame
    df = pd.DataFrame([{**{'model': model_name}, **metrics_dict}])
    
    # Append or create
    if metrics_file.exists():
        existing = pd.read_csv(metrics_file)
        df = pd.concat([existing, df], ignore_index=True)
    
    df.to_csv(metrics_file, index=False)
    print(f"Saved metrics to {metrics_file}")
    return metrics_file

def compare_models(metrics_file='outputs/metrics.csv', save_path='outputs/model_comparison.png'):
    """
    Create bar plot comparing model metrics.
    """
    if not Path(metrics_file).exists():
        print(f"Metrics file {metrics_file} not found.")
        return
    
    df = pd.read_csv(metrics_file)
    
    # Select numeric columns for comparison
    metric_cols = [col for col in df.columns if col != 'model']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metric_cols[:4]):  # Plot first 4 metrics
        ax = axes[idx]
        df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False)
        ax.set_title(f'{metric.upper()}')
        ax.set_ylabel('Score')
        ax.set_xlabel('Model')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")
    plt.close()

