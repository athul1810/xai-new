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
    
    # Create TF-IDF features with better settings for small datasets
    # Use more features and include trigrams for better discrimination
    embedder = TFIDFEmbedder(max_features=10000, ngram_range=(1, 3))
    X_train_tfidf = embedder.fit_transform(X_train)
    X_val_tfidf = embedder.transform(X_val)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Non-zero features per sample: {X_train_tfidf.getnnz(axis=1).mean():.1f}")
    
    # Convert to DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_tfidf, label=y_train)
    dval = xgb.DMatrix(X_val_tfidf, label=y_val)
    
    # Parameters - optimized for better discrimination and generalization
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 8,  # Moderate depth for better generalization
        'learning_rate': 0.05,  # Slightly higher learning rate
        'subsample': 0.8,  # More data per tree
        'colsample_bytree': 0.8,  # More features per tree
        'min_child_weight': 1,  # Standard value for better generalization
        'gamma': 0.1,  # Small minimum loss reduction
        'reg_alpha': 0.1,  # Light L1 regularization
        'reg_lambda': 1.0,  # Moderate L2 regularization
        'scale_pos_weight': 1,  # Handle class imbalance if needed
        'tree_method': 'hist',  # Faster training
        'random_state': random_state,
        'n_jobs': -1
    }
    
    # Train with early stopping - use more rounds for small datasets
    evals = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2000,  # More rounds for small datasets
        evals=evals,
        early_stopping_rounds=100,  # More patience
        verbose_eval=50
    )
    
    # Check prediction variance
    train_preds = model.predict(dtrain)
    val_preds = model.predict(dval)
    print(f"\nPrediction variance - Train: {np.var(train_preds):.6f}, Val: {np.var(val_preds):.6f}")
    print(f"Prediction range - Train: [{train_preds.min():.4f}, {train_preds.max():.4f}], Val: [{val_preds.min():.4f}, {val_preds.max():.4f}]")
    
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
    if xgb_results['feature_importance']:
        importance_df = pd.DataFrame([
            {'feature': k, 'importance': v} 
            for k, v in xgb_results['feature_importance'].items()
        ])
        if len(importance_df) > 0:
            importance_df = importance_df.sort_values('importance', ascending=False)
            importance_df.to_csv('outputs/xgb_feature_importance.csv', index=False)
        else:
            print("No feature importance data available (likely due to small dataset)")
    else:
        print("No feature importance data available (likely due to small dataset)")
    
    # Save metrics
    save_metrics(xgb_results['metrics'], 'XGBoost')
    plot_confusion_matrix(y_val, xgb_results['predictions'],
                         save_path='outputs/xgb_confusion_matrix.png')
    
    print("\nXGBoost model trained and saved!")

if __name__ == "__main__":
    main()

