"""
SHAP and LIME explainability for model predictions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# Optional imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

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

class ModelExplainer:
    """Wrapper for SHAP and LIME explanations."""
    
    def __init__(self, model, embedder=None, model_type='xgb'):
        self.model = model
        self.embedder = embedder
        self.model_type = model_type
        
        if SHAP_AVAILABLE:
            if model_type == 'xgb' and XGBOOST_AVAILABLE:
                try:
                    # Try to initialize SHAP TreeExplainer
                    # Note: Some XGBoost models may have base_score as string, causing errors
                    self.explainer_shap = shap.TreeExplainer(model)
                except (ValueError, TypeError) as e:
                    # If SHAP fails due to base_score format, try using model predictions directly
                    try:
                        # Use Explainer with model predictions as workaround
                        # Get sample predictions to create explainer
                        if embedder:
                            sample_texts = ['sample text for shap initialization']
                            X_sample = embedder.transform(sample_texts)
                            dmatrix_sample = xgb.DMatrix(X_sample)
                            sample_preds = model.predict(dmatrix_sample)
                            
                            # Use KernelExplainer as fallback (slower but more robust)
                            def model_predict(X):
                                dmatrix = xgb.DMatrix(X)
                                return model.predict(dmatrix)
                            
                            # Use a simpler explainer that doesn't need tree structure
                            self.explainer_shap = shap.KernelExplainer(
                                model_predict,
                                X_sample,
                                feature_names=embedder.get_feature_names() if hasattr(embedder, 'get_feature_names') else None
                            )
                            print("Using SHAP KernelExplainer (slower but more robust)")
                        else:
                            self.explainer_shap = None
                    except Exception as e2:
                        # If all SHAP methods fail, skip it
                        print(f"Warning: SHAP initialization failed: {e}. Fallback also failed: {e2}")
                        print("LIME explanations will still be available.")
                        self.explainer_shap = None
            elif model_type == 'bert':
                self.explainer_shap = None  # SHAP for transformers is more complex
            else:
                try:
                    # For linear models (SVM, RF)
                    sample_X = embedder.transform(['sample text']) if embedder else None
                    if sample_X is not None:
                        self.explainer_shap = shap.LinearExplainer(model, sample_X)
                    else:
                        self.explainer_shap = None
                except:
                    self.explainer_shap = None
        else:
            self.explainer_shap = None
    
    def explain_shap(self, texts, max_evals=100, save_path=None):
        """
        Generate SHAP explanations.
        
        Args:
            texts: List of texts to explain
            max_evals: Maximum evaluations for SHAP
            save_path: Path to save plots
        """
        if not SHAP_AVAILABLE:
            print("SHAP not available. Install with: pip install shap")
            return None
        
        if self.model_type == 'xgb' and XGBOOST_AVAILABLE:
            # Transform texts
            X = self.embedder.transform(texts)
            
            # Get SHAP values
            shap_values = self.explainer_shap.shap_values(X)
            
            # Plot
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                shap.summary_plot(shap_values, X, show=False, max_display=20)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved SHAP plot to {save_path}")
            
            return shap_values
        
        else:
            print(f"SHAP not implemented for {self.model_type}")
            return None
    
    def explain_lime(self, text, num_features=10, save_path=None):
        """
        Generate LIME explanation for a single text.
        
        Args:
            text: Text to explain
            num_features: Number of top features to show
            save_path: Path to save explanation
        """
        if not LIME_AVAILABLE:
            print("LIME not available. Install with: pip install lime")
            return None
        
        def predict_proba_wrapper(texts):
            """Wrapper for model prediction."""
            try:
                # Ensure texts is a list
                if isinstance(texts, str):
                    texts = [texts]
                elif not isinstance(texts, (list, tuple)):
                    texts = [str(texts)]
                
                if self.model_type == 'bert' and TRANSFORMERS_AVAILABLE:
                    # BERT model prediction
                    import torch
                    tokenizer = getattr(self, 'tokenizer', None)
                    if tokenizer is None:
                        # Try to get tokenizer from model if available
                        return np.array([[0.5, 0.5]] * len(texts))
                    
                    # Determine device
                    if torch.backends.mps.is_available():
                        device = 'mps'
                    elif torch.cuda.is_available():
                        device = 'cuda'
                    else:
                        device = 'cpu'
                    
                    self.model.to(device)
                    self.model.eval()
                    
                    probs_list = []
                    with torch.no_grad():
                        for text in texts:
                            encoded = tokenizer(
                                text,
                                padding=True,
                                truncation=True,
                                max_length=128,
                                return_tensors='pt'
                            )
                            encoded = {k: v.to(device) for k, v in encoded.items()}
                            outputs = self.model(**encoded)
                            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
                            probs_list.append(probs)
                    
                    return np.array(probs_list)
                elif self.model_type == 'xgb' and XGBOOST_AVAILABLE:
                    # Transform texts to features
                    X = self.embedder.transform(texts)
                    dmatrix = xgb.DMatrix(X)
                    preds = self.model.predict(dmatrix)
                    
                    # Ensure preds is numpy array
                    preds = np.array(preds).flatten()
                    
                    # XGBoost with binary:logistic outputs probabilities directly (0-1)
                    # Ensure we have proper shape
                    if len(preds.shape) == 0:
                        preds = np.array([preds])
                    
                    # Convert to 2D probability array [prob_class_0, prob_class_1]
                    # XGBoost outputs probability of positive class
                    probs = np.column_stack([1 - preds, preds])
                    
                    # Ensure probabilities sum to 1 and are valid
                    probs = np.clip(probs, 0, 1)
                    row_sums = probs.sum(axis=1, keepdims=True)
                    probs = probs / row_sums
                    
                    return probs
                else:
                    # For other models (SVM, RF), use predict_proba
                    if self.embedder:
                        X = self.embedder.transform(texts)
                        probs = self.model.predict_proba(X)
                        return probs
                    else:
                        # Fallback for models without embedder
                        return np.array([[0.5, 0.5]] * len(texts))
            except Exception as e:
                # Return neutral probabilities on error
                num_texts = len(texts) if isinstance(texts, (list, tuple)) else 1
                if num_texts == 0:
                    num_texts = 1
                return np.array([[0.5, 0.5]] * num_texts)
        
        explainer = LimeTextExplainer(class_names=['Low Risk', 'High Risk'])
        
        # Test if model has variation in predictions
        test_texts = [text, text + " extra", text[:len(text)//2] if len(text) > 10 else text + " test"]
        test_probs = predict_proba_wrapper(test_texts)
        prob_variance = np.var(test_probs[:, 1]) if len(test_probs.shape) > 1 else np.var(test_probs)
        
        # Increase num_samples significantly if model has low variance
        # Use more samples and distance weighting to detect subtle differences
        num_samples = 10000 if prob_variance < 0.001 else 5000
        
        # Use distance_metric='cosine' to better handle text similarities
        explanation = explainer.explain_instance(
            text,
            predict_proba_wrapper,
            num_features=num_features,
            num_samples=num_samples,
            top_labels=1,
            distance_metric='cosine'  # Better for text similarity
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            explanation.save_to_file(save_path)
            print(f"Saved LIME explanation to {save_path}")
        
        return explanation

def generate_shap_explanations(model_path, embedder_path, texts, output_dir='outputs/shap_plots'):
    """Generate SHAP explanations for XGBoost model."""
    if not XGBOOST_AVAILABLE:
        print("XGBoost not available. Cannot generate SHAP explanations.")
        return None
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from features.embeddings import TFIDFEmbedder
    
    # Load model
    model = xgb.Booster()
    model.load_model(model_path)
    
    embedder = TFIDFEmbedder()
    embedder.load(embedder_path)
    
    # Create explainer
    explainer = ModelExplainer(model, embedder, model_type='xgb')
    
    # Generate explanations
    shap_values = explainer.explain_shap(
        texts,
        save_path=f'{output_dir}/shap_summary.png'
    )
    
    return shap_values

def generate_lime_explanations(model_path, embedder_path, texts, output_dir='outputs/lime_plots'):
    """Generate LIME explanations."""
    if not LIME_AVAILABLE:
        print("LIME not available. Cannot generate LIME explanations.")
        return None
    
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from features.embeddings import TFIDFEmbedder
    
    # Try to load as XGBoost first, fallback to joblib
    try:
        if XGBOOST_AVAILABLE and model_path.endswith('.json'):
            model = xgb.Booster()
            model.load_model(model_path)
        else:
            import joblib
            model = joblib.load(model_path)
    except:
        import joblib
        model = joblib.load(model_path)
    
    embedder = TFIDFEmbedder()
    embedder.load(embedder_path)
    
    # Create explainer
    explainer = ModelExplainer(model, embedder, model_type='xgb')
    
    # Generate explanations for each text
    explanations = []
    for idx, text in enumerate(texts):
        exp = explainer.explain_lime(
            text,
            num_features=10,
            save_path=f'{output_dir}/lime_explanation_{idx}.html'
        )
        explanations.append(exp)
    
    return explanations

if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from features.embeddings import TFIDFEmbedder
    
    # Load sample data
    test_df = pd.read_csv('data/test.csv')
    sample_texts = test_df['text'].head(5).tolist()
    
    # Generate SHAP explanations
    print("Generating SHAP explanations...")
    generate_shap_explanations(
        'trained_models/xgb_model.json',
        'trained_models/xgb_tfidf.pkl',
        sample_texts
    )
    
    # Generate LIME explanations
    print("Generating LIME explanations...")
    generate_lime_explanations(
        'trained_models/xgb_model.json',
        'trained_models/xgb_tfidf.pkl',
        sample_texts
    )

