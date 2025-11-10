"""
Ensemble stacking: Combine XGBoost and BERT predictions using Logistic Regression.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# Optional imports
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from features.embeddings import TFIDFEmbedder
from utils.metrics import compute_metrics, plot_confusion_matrix, save_metrics

def load_xgb_model(model_path='trained_models/xgb_model.json', 
                   embedder_path='trained_models/xgb_tfidf.pkl'):
    """Load trained XGBoost model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available. Install libomp: brew install libomp")
    
    model = xgb.Booster()
    model.load_model(model_path)
    
    embedder = TFIDFEmbedder()
    embedder.load(embedder_path)
    
    return model, embedder

def load_bert_model(model_path='trained_models/bert'):
    """Load trained BERT model."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers not available. Install with: pip install transformers torch")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    
    return model, tokenizer

def get_xgb_predictions(model, embedder, texts):
    """Get XGBoost predictions."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost not available")
    X_tfidf = embedder.transform(texts)
    dmatrix = xgb.DMatrix(X_tfidf)
    return model.predict(dmatrix)

def get_bert_predictions(model, tokenizer, texts, device='cpu', batch_size=16):
    """Get BERT predictions."""
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers not available")
    
    import torch
    model.to(device)
    predictions = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            predictions.extend(probs)
    
    return np.array(predictions)

def train_stacked_ensemble(X_train, y_train, X_val, y_val, random_state=42):
    """Train stacked ensemble model."""
    print("Training stacked ensemble (XGBoost + BERT)...")
    
    # Load base models
    print("Loading XGBoost model...")
    xgb_model, xgb_embedder = load_xgb_model()
    
    print("Loading BERT model...")
    bert_model, bert_tokenizer = load_bert_model()
    
    # Get base model predictions
    print("Generating base model predictions...")
    xgb_preds_train = get_xgb_predictions(xgb_model, xgb_embedder, X_train)
    bert_preds_train = get_bert_predictions(bert_model, bert_tokenizer, X_train)
    
    xgb_preds_val = get_xgb_predictions(xgb_model, xgb_embedder, X_val)
    bert_preds_val = get_bert_predictions(bert_model, bert_tokenizer, X_val)
    
    # Stack features
    X_stack_train = np.column_stack([xgb_preds_train, bert_preds_train])
    X_stack_val = np.column_stack([xgb_preds_val, bert_preds_val])
    
    # Train meta-learner (Logistic Regression)
    print("Training meta-learner...")
    meta_learner = LogisticRegression(random_state=random_state, max_iter=1000)
    meta_learner.fit(X_stack_train, y_train)
    
    # Predictions
    y_pred = meta_learner.predict(X_stack_val)
    y_proba = meta_learner.predict_proba(X_stack_val)[:, 1]
    
    # Metrics
    metrics = compute_metrics(y_val, y_pred, y_proba)
    print(f"Stacked Ensemble Metrics: {metrics}")
    
    return {
        'xgb_model': xgb_model,
        'xgb_embedder': xgb_embedder,
        'bert_model': bert_model,
        'bert_tokenizer': bert_tokenizer,
        'meta_learner': meta_learner,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_proba
    }

def predict_stacked(texts, xgb_model, xgb_embedder, bert_model, bert_tokenizer, meta_learner):
    """Make predictions with stacked ensemble."""
    xgb_preds = get_xgb_predictions(xgb_model, xgb_embedder, texts)
    bert_preds = get_bert_predictions(bert_model, bert_tokenizer, texts)
    
    X_stack = np.column_stack([xgb_preds, bert_preds])
    y_pred = meta_learner.predict(X_stack)
    y_proba = meta_learner.predict_proba(X_stack)[:, 1]
    
    return y_pred, y_proba

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
    
    # Train stacked ensemble
    stack_results = train_stacked_ensemble(X_train, y_train, X_val, y_val)
    
    # Save meta-learner
    joblib.dump(stack_results['meta_learner'], 'trained_models/stack_meta_learner.pkl')
    
    # Save metrics
    save_metrics(stack_results['metrics'], 'Stacked_Ensemble')
    plot_confusion_matrix(y_val, stack_results['predictions'],
                         save_path='outputs/stack_confusion_matrix.png')
    
    print("\nStacked ensemble model trained and saved!")

if __name__ == "__main__":
    main()

