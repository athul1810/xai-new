"""
Train XGBoost classifier with TF-IDF features and early stopping.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from features.embeddings import TFIDFEmbedder
from utils.metrics import compute_metrics, plot_confusion_matrix, save_metrics

def train_xgboost(X_train, y_train, X_val, y_val, random_state=42):
    """Train XGBoost classifier with early stopping."""
    print("Training XGBoost...")
    
    # Create TF-IDF features
    embedder = TFIDFEmbedder(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = embedder.fit_transform(X_train)
    X_val_tfidf = embedder.transform(X_val)
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
    dval = xgb.DMatrix(X_val_tfidf, label=y_val)
    
    # Parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': random_state,
        'n_jobs': -1
    }
    
    # Train with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )
    
    # Predictions
    y_pred = (model.predict(dval) > 0.5).astype(int)
    y_proba = model.predict(dval)
    
    # Metrics
    metrics = compute_metrics(y_val, y_pred, y_proba)
    print(f"XGBoost Metrics: {metrics}")
    
    # Feature importance
    feature_importance = model.get_score(importance_type='gain')
    print(f"\nTop 10 Feature Importances:")
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, importance in sorted_features:
        print(f"  {feat}: {importance:.4f}")
    
    return {
        'model': model,
        'embedder': embedder,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_proba,
        'feature_importance': feature_importance
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
    
    # Train XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Save model
    xgb_results['model'].save_model('trained_models/xgb_model.json')
    xgb_results['embedder'].save('trained_models/xgb_tfidf.pkl')
    
    # Save feature importance
    importance_df = pd.DataFrame([
        {'feature': k, 'importance': v} 
        for k, v in xgb_results['feature_importance'].items()
    ]).sort_values('importance', ascending=False)
    importance_df.to_csv('outputs/xgb_feature_importance.csv', index=False)
    
    # Save metrics
    save_metrics(xgb_results['metrics'], 'XGBoost')
    plot_confusion_matrix(y_val, xgb_results['predictions'],
                         save_path='outputs/xgb_confusion_matrix.png')
    
    print("\nXGBoost model trained and saved!")

if __name__ == "__main__":
    main()

