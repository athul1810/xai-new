"""
Streamlit web app for substance-abuse-risk detection with explainability and Ollama chat.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import requests
import sys
from pathlib import Path
import shap
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from features.embeddings import TFIDFEmbedder
from explain.shap_lime import ModelExplainer
from models.stack import load_xgb_model, load_bert_model, predict_stacked

# Page config
st.set_page_config(
    page_title="XAI Substance-Abuse-Risk Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    
    try:
        # XGBoost
        xgb_model = xgb.Booster()
        xgb_model.load_model('trained_models/xgb_model.json')
        xgb_embedder = TFIDFEmbedder()
        xgb_embedder.load('trained_models/xgb_tfidf.pkl')
        models['xgb'] = {'model': xgb_model, 'embedder': xgb_embedder}
    except Exception as e:
        st.warning(f"XGBoost model not found: {e}")
    
    try:
        # BERT
        bert_model, bert_tokenizer = load_bert_model()
        models['bert'] = {'model': bert_model, 'tokenizer': bert_tokenizer}
    except Exception as e:
        st.warning(f"BERT model not found: {e}")
    
    try:
        # Stacked Ensemble
        meta_learner = joblib.load('trained_models/stack_meta_learner.pkl')
        models['stack'] = {'meta_learner': meta_learner}
    except Exception as e:
        st.warning(f"Stacked ensemble not found: {e}")
    
    return models

def query_ollama(prompt, model="llama3"):
    """
    Query Ollama API for natural language explanations.
    
    Args:
        prompt: Text prompt for Ollama
        model: Model name (default: llama3)
    
    Returns:
        str: Response from Ollama
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", "No response from Ollama.")
        else:
            return f"Error: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Ollama is not running. Please start Ollama with: `ollama serve`"
    except Exception as e:
        return f"Error querying Ollama: {str(e)}"

def predict_with_xgb(text, models):
    """Make prediction with XGBoost."""
    if 'xgb' not in models:
        return None, None
    
    xgb_model = models['xgb']['model']
    embedder = models['xgb']['embedder']
    
    X = embedder.transform([text])
    dmatrix = xgb.DMatrix(X)
    proba = xgb_model.predict(dmatrix)[0]
    pred = 1 if proba > 0.5 else 0
    
    return pred, proba

def predict_with_bert(text, models):
    """Make prediction with BERT."""
    if 'bert' not in models:
        return None, None
    
    model = models['bert']['model']
    tokenizer = models['bert']['tokenizer']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.eval()
    
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors='pt'
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    with torch.no_grad():
        outputs = model(**encoded)
        probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()
        proba = probs[1]
        pred = 1 if proba > 0.5 else 0
    
    return pred, proba

def predict_with_stack(text, models):
    """Make prediction with stacked ensemble."""
    if 'stack' not in models or 'xgb' not in models or 'bert' not in models:
        return None, None
    
    xgb_model = models['xgb']['model']
    xgb_embedder = models['xgb']['embedder']
    bert_model = models['bert']['model']
    bert_tokenizer = models['bert']['tokenizer']
    meta_learner = models['stack']['meta_learner']
    
    # Get base predictions
    xgb_pred = get_xgb_predictions(xgb_model, xgb_embedder, [text])[0]
    bert_pred = get_bert_predictions(bert_model, bert_tokenizer, [text])[0]
    
    # Stack and predict
    X_stack = np.array([[xgb_pred, bert_pred]])
    pred = meta_learner.predict(X_stack)[0]
    proba = meta_learner.predict_proba(X_stack)[0][1]
    
    return pred, proba

def get_xgb_predictions(model, embedder, texts):
    """Helper for XGBoost predictions."""
    X = embedder.transform(texts)
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)

def get_bert_predictions(model, tokenizer, texts, device='cpu'):
    """Helper for BERT predictions."""
    model.to(device)
    predictions = []
    
    encoded = tokenizer(
        texts,
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

# Main app
def main():
    st.title("🔍 Explainable AI: Substance-Abuse-Risk Detection")
    st.markdown("---")
    
    # Load models
    with st.spinner("Loading models..."):
        models = load_models()
    
    if not models:
        st.error("No models found! Please train models first.")
        return
    
    # Sidebar
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose model:",
        ['XGBoost', 'BERT', 'Stacked Ensemble'],
        index=0
    )
    
    st.sidebar.markdown("---")
    st.sidebar.header("Ollama Settings")
    use_ollama = st.sidebar.checkbox("Enable Ollama explanations", value=False)
    ollama_model = st.sidebar.selectbox("Ollama Model", ["llama3", "llama2", "mistral"], index=0)
    
    # Main input
    st.header("Input Text")
    user_text = st.text_area(
        "Enter text to analyze:",
        height=150,
        placeholder="Example: I've been struggling with substance use and need help..."
    )
    
    if st.button("Analyze", type="primary"):
        if not user_text.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        # Predictions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction")
            
            if model_choice == 'XGBoost':
                pred, proba = predict_with_xgb(user_text, models)
            elif model_choice == 'BERT':
                pred, proba = predict_with_bert(user_text, models)
            else:  # Stacked Ensemble
                pred, proba = predict_with_stack(user_text, models)
            
            if pred is not None:
                risk_level = "🔴 High Risk" if pred == 1 else "🟢 Low Risk"
                st.markdown(f"### {risk_level}")
                st.progress(proba)
                st.metric("Risk Probability", f"{proba:.2%}")
            else:
                st.error("Prediction failed. Model not available.")
        
        with col2:
            st.subheader("Explanation")
            
            if model_choice == 'XGBoost' and 'xgb' in models:
                # SHAP explanation
                try:
                    explainer = ModelExplainer(
                        models['xgb']['model'],
                        models['xgb']['embedder'],
                        model_type='xgb'
                    )
                    
                    # Get SHAP values
                    X = models['xgb']['embedder'].transform([user_text])
                    shap_values = explainer.explain_shap([user_text])
                    
                    if shap_values is not None:
                        st.markdown("**SHAP Feature Importance:**")
                        # Create summary plot
                        fig, ax = plt.subplots()
                        shap.summary_plot(shap_values, X, show=False, max_display=10)
                        st.pyplot(fig)
                        
                        # LIME explanation
                        lime_exp = explainer.explain_lime(user_text, num_features=10)
                        st.markdown("**LIME Top Features:**")
                        for feature, weight in lime_exp.as_list():
                            st.write(f"- {feature}: {weight:.4f}")
                
                except Exception as e:
                    st.error(f"Explanation error: {e}")
        
        # Ollama explanation
        if use_ollama:
            st.markdown("---")
            st.subheader("🤖 Natural Language Explanation (Ollama)")
            
            with st.spinner("Generating explanation..."):
                # Build prompt
                top_features = "substance, use, help, struggling"  # Simplified
                prompt = f"""You are an Explainable AI assistant helping to explain a machine learning model's prediction.

Model Prediction: {'High Risk' if pred == 1 else 'Low Risk'} (Probability: {proba:.2%})
Input Text: "{user_text}"
Top Features: {top_features}

Explain the reasoning behind this classification in simple, empathetic language. Focus on why the model made this prediction based on the text content."""
                
                ollama_response = query_ollama(prompt, ollama_model)
                st.markdown(ollama_response)
        
        # Model comparison (if multiple models available)
        if len(models) > 1:
            st.markdown("---")
            st.subheader("Model Comparison")
            
            comparison_data = []
            for name, func in [('XGBoost', predict_with_xgb), ('BERT', predict_with_bert)]:
                if name.lower().replace(' ', '_') in models:
                    p, prob = func(user_text, models)
                    if p is not None:
                        comparison_data.append({
                            'Model': name,
                            'Prediction': 'High Risk' if p == 1 else 'Low Risk',
                            'Probability': f"{prob:.2%}"
                        })
            
            if comparison_data:
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

if __name__ == "__main__":
    main()

