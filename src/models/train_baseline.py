"""
Train baseline models: SVM and RandomForest with TF-IDF features.
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from features.embeddings import TFIDFEmbedder
from utils.metrics import compute_metrics, plot_confusion_matrix, save_metrics

def train_svm(X_train, y_train, X_val, y_val, random_state=42):
    """Train SVM classifier."""
    print("Training SVM...")
    
    # Create pipeline with more features and better ngrams
    embedder = TFIDFEmbedder(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = embedder.fit_transform(X_train)
    X_val_tfidf = embedder.transform(X_val)
    
    # Train SVM with better hyperparameters
    # Use linear kernel for better generalization, or RBF with tuned C
    svm = SVC(kernel='linear', probability=True, random_state=random_state, C=0.1)
    svm.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = svm.predict(X_val_tfidf)
    y_proba = svm.predict_proba(X_val_tfidf)[:, 1]
    
    # Metrics
    metrics = compute_metrics(y_val, y_pred, y_proba)
    print(f"SVM Metrics: {metrics}")
    
    return {
        'model': svm,
        'embedder': embedder,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_proba
    }

def train_random_forest(X_train, y_train, X_val, y_val, random_state=42, n_estimators=200):
    """Train RandomForest classifier."""
    print("Training RandomForest...")
    
    # Create pipeline with more features
    embedder = TFIDFEmbedder(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = embedder.fit_transform(X_train)
    X_val_tfidf = embedder.transform(X_val)
    
    # Train RandomForest with better hyperparameters for generalization
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=15,  # Reduce depth to prevent overfitting
        min_samples_split=5,  # Require more samples to split
        min_samples_leaf=2,  # Require minimum samples in leaf
        max_features='sqrt',  # Use sqrt of features for better generalization
        class_weight='balanced',  # Handle class imbalance
        random_state=random_state,
        n_jobs=-1
    )
    rf.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = rf.predict(X_val_tfidf)
    y_proba = rf.predict_proba(X_val_tfidf)[:, 1]
    
    # Metrics
    metrics = compute_metrics(y_val, y_pred, y_proba)
    print(f"RandomForest Metrics: {metrics}")
    
    return {
        'model': rf,
        'embedder': embedder,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_proba
    }

def main():
    """Main training pipeline."""
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    X_val = val_df['text'].values
    y_val = val_df['label'].values
    
    # Create output directory
    Path('trained_models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Train SVM
    svm_results = train_svm(X_train, y_train, X_val, y_val)
    
    # Save SVM
    joblib.dump(svm_results['model'], 'trained_models/svm_model.pkl')
    svm_results['embedder'].save('trained_models/svm_tfidf.pkl')
    save_metrics(svm_results['metrics'], 'SVM')
    plot_confusion_matrix(y_val, svm_results['predictions'], 
                         save_path='outputs/svm_confusion_matrix.png')
    
    # Train RandomForest
    rf_results = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Save RandomForest
    joblib.dump(rf_results['model'], 'trained_models/rf_model.pkl')
    rf_results['embedder'].save('trained_models/rf_tfidf.pkl')
    save_metrics(rf_results['metrics'], 'RandomForest')
    plot_confusion_matrix(y_val, rf_results['predictions'],
                         save_path='outputs/rf_confusion_matrix.png')
    
    print("\nBaseline models trained and saved!")

if __name__ == "__main__":
    main()

