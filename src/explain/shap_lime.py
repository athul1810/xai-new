"""
SHAP and LIME explainability for model predictions.
"""

import numpy as np
import pandas as pd
import shap
import lime
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelExplainer:
    """Wrapper for SHAP and LIME explanations."""
    
    def __init__(self, model, embedder=None, model_type='xgb'):
        self.model = model
        self.embedder = embedder
        self.model_type = model_type
        
        if model_type == 'xgb':
            self.explainer_shap = shap.TreeExplainer(model)
        elif model_type == 'bert':
            self.explainer_shap = None  # SHAP for transformers is more complex
        else:
            self.explainer_shap = shap.LinearExplainer(model, embedder.transform(['']))
    
    def explain_shap(self, texts, max_evals=100, save_path=None):
        """
        Generate SHAP explanations.
        
        Args:
            texts: List of texts to explain
            max_evals: Maximum evaluations for SHAP
            save_path: Path to save plots
        """
        if self.model_type == 'xgb':
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
        def predict_proba_wrapper(texts):
            """Wrapper for model prediction."""
            if self.model_type == 'xgb':
                X = self.embedder.transform(texts)
                dmatrix = xgb.DMatrix(X)
                preds = self.model.predict(dmatrix)
                # Convert to probabilities
                probs = np.column_stack([1 - preds, preds])
                return probs
            else:
                # For other models, implement accordingly
                return np.array([[0.5, 0.5]] * len(texts))
        
        explainer = LimeTextExplainer(class_names=['Low Risk', 'High Risk'])
        explanation = explainer.explain_instance(
            text,
            predict_proba_wrapper,
            num_features=num_features,
            top_labels=1
        )
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            explanation.save_to_file(save_path)
            print(f"Saved LIME explanation to {save_path}")
        
        return explanation

def generate_shap_explanations(model_path, embedder_path, texts, output_dir='outputs/shap_plots'):
    """Generate SHAP explanations for XGBoost model."""
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

