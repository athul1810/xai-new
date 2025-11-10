"""
Streamlit web app for substance-abuse-risk detection with explainability and ChatGPT chat.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
import sys
from pathlib import Path
import matplotlib.pyplot as plt

# OpenAI import
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional imports with graceful handling
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception as e:
    XGBOOST_AVAILABLE = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from features.embeddings import TFIDFEmbedder
from explain.shap_lime import ModelExplainer, LIME_AVAILABLE

# Optional stack imports (only if XGBoost/BERT available)
try:
    from models.stack import load_xgb_model, load_bert_model, predict_stacked
    STACK_AVAILABLE = True
except Exception:
    STACK_AVAILABLE = False
    load_xgb_model = None
    load_bert_model = None
    predict_stacked = None

# Page config
st.set_page_config(
    page_title="XAI Substance-Abuse-Risk Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global font */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main styling */
    .main {
        padding: 2rem 2rem;
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }
    
    /* Header styling */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-align: center;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 50%, #d299c2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        text-shadow: 0 0 30px rgba(168, 237, 234, 0.3);
    }
    
    h2 {
        font-family: 'Poppins', sans-serif;
        color: #e0e0e0;
        font-size: 1.9rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid transparent;
        border-image: linear-gradient(90deg, #a8edea 0%, #fed6e3 100%) 1;
        padding-bottom: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    h3 {
        font-family: 'Poppins', sans-serif;
        color: #d0d0d0;
        font-size: 1.5rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    
    /* Body text */
    p, div, span {
        color: #e8e8e8;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1625 0%, #2d1b3d 100%);
    }
    
    [data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Card-like containers */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), 0 0 0 1px rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .explanation-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(168, 237, 234, 0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.3);
    }
    
    /* Risk level badges */
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
        color: white;
        padding: 1.2rem 2.5rem;
        border-radius: 30px;
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        box-shadow: 0 10px 40px rgba(255, 107, 107, 0.5), 0 0 0 1px rgba(255,255,255,0.1);
        margin: 1rem 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.02); }
    }
    
    .risk-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #00d4ff 100%);
        color: white;
        padding: 1.2rem 2.5rem;
        border-radius: 30px;
        font-size: 1.6rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        text-align: center;
        box-shadow: 0 10px 40px rgba(79, 172, 254, 0.5), 0 0 0 1px rgba(255,255,255,0.1);
        margin: 1rem 0;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%) !important;
        color: #1a1625 !important;
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        font-family: 'Poppins', sans-serif !important;
        padding: 0.9rem 2.5rem !important;
        border-radius: 30px !important;
        border: none !important;
        box-shadow: 0 8px 30px rgba(168, 237, 234, 0.4), 0 0 0 1px rgba(255,255,255,0.1) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.02em !important;
        -webkit-text-fill-color: #1a1625 !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02) !important;
        box-shadow: 0 12px 40px rgba(168, 237, 234, 0.6), 0 0 0 1px rgba(255,255,255,0.2) !important;
        background: linear-gradient(135deg, #b8f5f2 0%, #ffd6e3 100%) !important;
        color: #1a1625 !important;
        -webkit-text-fill-color: #1a1625 !important;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98) !important;
    }
    
    /* Force button text to be dark and visible */
    .stButton > button > div > p,
    .stButton > button > div,
    .stButton > button span {
        color: #1a1625 !important;
        -webkit-text-fill-color: #1a1625 !important;
        font-weight: 700 !important;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 16px;
        border: 2px solid rgba(168, 237, 234, 0.2);
        background: rgba(255, 255, 255, 0.08) !important;
        color: #ffffff !important;
        padding: 1.2rem;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #a8edea;
        box-shadow: 0 0 0 4px rgba(168, 237, 234, 0.2), 0 8px 32px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
        outline: none;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: rgba(255, 255, 255, 0.5) !important;
    }
    
    /* Ensure textarea text is always visible */
    .stTextArea textarea {
        color: #ffffff !important;
    }
    
    /* Override any Streamlit default text colors */
    textarea {
        color: #ffffff !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 50%, #d299c2 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(168, 237, 234, 0.3);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem;
        font-weight: 500;
        color: rgba(232, 232, 232, 0.7);
        font-family: 'Inter', sans-serif;
    }
    
    /* Feature list styling */
    .feature-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        border-radius: 12px;
        border-left: 4px solid #a8edea;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(168, 237, 234, 0.2);
    }
    
    .feature-item:hover {
        background: rgba(255, 255, 255, 0.1);
        transform: translateX(8px) scale(1.02);
        box-shadow: 0 4px 20px rgba(168, 237, 234, 0.2);
        border-color: rgba(168, 237, 234, 0.4);
    }
    
    /* Comparison table styling */
    .dataframe {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(168, 237, 234, 0.2);
    }
    
    .dataframe thead {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #1a1625;
        font-weight: 700;
        font-family: 'Poppins', sans-serif;
    }
    
    .dataframe tbody tr {
        background: rgba(255, 255, 255, 0.03);
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Info boxes */
    .stInfo {
        border-radius: 12px;
        border-left: 5px solid #a8edea;
        background: rgba(168, 237, 234, 0.1);
        border: 1px solid rgba(168, 237, 234, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        border-radius: 12px;
        border-left: 5px solid #ffd93d;
        background: rgba(255, 217, 61, 0.1);
        border: 1px solid rgba(255, 217, 61, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stError {
        border-radius: 12px;
        border-left: 5px solid #ff6b6b;
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(168, 237, 234, 0.2);
        border-radius: 12px;
        color: #e8e8e8;
        backdrop-filter: blur(10px);
    }
    
    .stSelectbox > div > div > div:hover {
        border-color: rgba(168, 237, 234, 0.4);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #e8e8e8;
        font-weight: 500;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #a8edea;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label {
        color: #e8e8e8 !important;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #b8f5f2 0%, #ffd6e3 100%);
    }
    
    /* Force white text in all input fields */
    input[type="text"],
    input[type="textarea"],
    textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }
    
    /* Ensure label text is visible */
    label {
        color: #e8e8e8 !important;
    }
    
    /* Make sure all text in the app is visible */
    .stTextInput > div > div > input,
    .stTextInput > div > div > input:focus {
        color: #ffffff !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all trained models."""
    models = {}
    
    # SVM
    try:
        svm_model = joblib.load('trained_models/svm_model.pkl')
        svm_embedder = TFIDFEmbedder()
        svm_embedder.load('trained_models/svm_tfidf.pkl')
        models['svm'] = {'model': svm_model, 'embedder': svm_embedder}
    except Exception as e:
        pass
    
    # RandomForest
    try:
        rf_model = joblib.load('trained_models/rf_model.pkl')
        rf_embedder = TFIDFEmbedder()
        rf_embedder.load('trained_models/rf_tfidf.pkl')
        models['rf'] = {'model': rf_model, 'embedder': rf_embedder}
    except Exception as e:
        pass
    
    # XGBoost (if available)
    if XGBOOST_AVAILABLE:
        try:
            xgb_model = xgb.Booster()
            xgb_model.load_model('trained_models/xgb_model.json')
            xgb_embedder = TFIDFEmbedder()
            xgb_embedder.load('trained_models/xgb_tfidf.pkl')
            models['xgb'] = {'model': xgb_model, 'embedder': xgb_embedder}
        except Exception as e:
            pass
    
    # BERT (if available)
    if TRANSFORMERS_AVAILABLE and STACK_AVAILABLE and load_bert_model:
        try:
            bert_model, bert_tokenizer = load_bert_model()
            models['bert'] = {'model': bert_model, 'tokenizer': bert_tokenizer}
        except Exception as e:
            pass
    
    # Stacked Ensemble
    try:
        meta_learner = joblib.load('trained_models/stack_meta_learner.pkl')
        models['stack'] = {'meta_learner': meta_learner}
    except Exception as e:
        pass
    
    return models

def query_chatgpt(user_message, api_key, model="gpt-3.5-turbo", chat_history=None):
    """
    Query ChatGPT API for chat responses.
    
    Args:
        user_message: User's message
        api_key: OpenAI API key
        model: Model name (default: gpt-3.5-turbo)
        chat_history: List of previous messages in format [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        str: Response from ChatGPT
    """
    if not OPENAI_AVAILABLE:
        return "⚠️ OpenAI library not installed. Install with: pip install openai"
    
    if not api_key or not api_key.strip():
        return "⚠️ Please enter your OpenAI API key in the sidebar."
    
    try:
        client = OpenAI(api_key=api_key.strip())
        
        # Build system message
        system_message = {
            "role": "system",
            "content": """You are a compassionate and helpful AI assistant specializing in mental health and substance abuse support. 
You provide empathetic, non-judgmental responses and helpful guidance. You never ask for names or give generic greetings. 
You respond directly to what the user says with understanding and care. Keep responses conversational and supportive."""
        }
        
        # Build messages list
        messages = [system_message]
        
        # Add chat history if available
        if chat_history and len(chat_history) > 0:
            # Convert chat history to OpenAI format (last 10 messages for context)
            for msg in chat_history[-10:]:
                messages.append({
                    "role": msg['role'],
                    "content": msg['content']
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        if response.choices and len(response.choices) > 0:
            return response.choices[0].message.content.strip()
        else:
            return "⚠️ No response from ChatGPT."
            
    except Exception as e:
        error_str = str(e)
        if "api_key" in error_str.lower() or "authentication" in error_str.lower() or "invalid" in error_str.lower():
            return "⚠️ Invalid API key. Please check your OpenAI API key in the sidebar."
        elif "rate limit" in error_str.lower():
            return "⚠️ Rate limit exceeded. Please wait a moment and try again."
        elif "insufficient_quota" in error_str.lower():
            return "⚠️ API quota exceeded. Please check your OpenAI account billing."
        else:
            return f"⚠️ Error: {error_str}"

def predict_with_svm(text, models):
    """Make prediction with SVM."""
    if 'svm' not in models:
        return None, None
    
    svm_model = models['svm']['model']
    embedder = models['svm']['embedder']
    
    X = embedder.transform([text])
    proba = svm_model.predict_proba(X)[0][1]
    proba = float(proba)  # Convert to Python float
    pred = svm_model.predict(X)[0]
    
    return pred, proba

def predict_with_rf(text, models):
    """Make prediction with RandomForest."""
    if 'rf' not in models:
        return None, None
    
    rf_model = models['rf']['model']
    embedder = models['rf']['embedder']
    
    X = embedder.transform([text])
    proba = rf_model.predict_proba(X)[0][1]
    proba = float(proba)  # Convert to Python float
    pred = rf_model.predict(X)[0]
    
    return pred, proba

def predict_with_xgb(text, models):
    """Make prediction with XGBoost."""
    if not XGBOOST_AVAILABLE or 'xgb' not in models:
        return None, None
    
    xgb_model = models['xgb']['model']
    embedder = models['xgb']['embedder']
    
    X = embedder.transform([text])
    dmatrix = xgb.DMatrix(X)
    proba = xgb_model.predict(dmatrix)[0]
    # Convert to Python float to avoid type issues with Streamlit
    proba = float(proba)
    pred = 1 if proba > 0.5 else 0
    
    return pred, proba

def predict_with_bert(text, models):
    """Make prediction with BERT."""
    if not TRANSFORMERS_AVAILABLE or 'bert' not in models:
        return None, None
    
    import torch
    
    try:
        model = models['bert']['model']
        tokenizer = models['bert']['tokenizer']
        
        # Determine device (MPS for Mac, CUDA for GPU, CPU otherwise)
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
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
            proba = float(probs[1])  # Convert to Python float
            pred = 1 if proba > 0.5 else 0
        
        return pred, proba
    except Exception as e:
        # Don't use st.error here as this might be called outside Streamlit context
        print(f"BERT prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def predict_with_stack(text, models):
    """Make prediction with stacked ensemble."""
    if not STACK_AVAILABLE or 'stack' not in models:
        return None, None
    
    # Check if base models are available
    if 'xgb' not in models or 'bert' not in models:
        return None, None
    
    try:
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
        proba = float(proba)  # Convert to Python float
        
        return pred, proba
    except Exception:
        return None, None

def get_xgb_predictions(model, embedder, texts):
    """Helper for XGBoost predictions."""
    X = embedder.transform(texts)
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)

def get_bert_predictions(model, tokenizer, texts, device=None):
    """Helper for BERT predictions."""
    import torch
    
    # Determine device if not specified
    if device is None:
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
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
    # Header with gradient
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0 2rem 0;">
        <h1>🔍 Explainable AI</h1>
        <h2 style="color: rgba(232, 232, 232, 0.8); font-size: 1.4rem; font-weight: 400; margin-top: -0.5rem; letter-spacing: 0.05em;">
            Substance-Abuse-Risk Detection System
        </h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        models = load_models()
    
    if not models:
        st.error("❌ No models found! Please train models first.")
        return
    
    # Sidebar with better styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); 
                padding: 1.5rem; border-radius: 16px; margin-bottom: 2rem; 
                box-shadow: 0 8px 32px rgba(168, 237, 234, 0.3);">
        <h2 style="color: #000000 !important; margin: 0; font-size: 1.6rem; font-weight: 700; font-family: 'Poppins', sans-serif; text-shadow: 0 1px 2px rgba(255,255,255,0.3);">⚙️ Settings</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 🤖 Model Selection")
    
    # Build available models list
    available_models = []
    if 'svm' in models:
        available_models.append('SVM')
    if 'rf' in models:
        available_models.append('RandomForest')
    if 'xgb' in models:
        available_models.append('XGBoost')
    if 'bert' in models:
        available_models.append('BERT')
    if 'stack' in models:
        available_models.append('Stacked Ensemble')
    
    if not available_models:
        st.error("No models available! Please train models first.")
        return
    
    model_choice = st.sidebar.selectbox(
        "Select AI Model:",
        available_models,
        index=0,
        help="Choose the machine learning model for risk detection"
    )
    
    # Model info badges
    model_info = {
        'SVM': '⚡ Fast & Linear',
        'RandomForest': '🌲 Ensemble Method',
        'XGBoost': '🚀 Gradient Boosting',
        'BERT': '🧠 Deep Learning',
        'Stacked Ensemble': '🎯 Best Performance'
    }
    if model_choice in model_info:
        st.sidebar.markdown(f"<p style='color: #7f8c8d; font-size: 0.9rem;'>{model_info[model_choice]}</p>", unsafe_allow_html=True)
    
    # Show info about optional features in sidebar
    if not XGBOOST_AVAILABLE or not TRANSFORMERS_AVAILABLE or not SHAP_AVAILABLE:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 💡 Optional Features")
        missing_features = []
        if not XGBOOST_AVAILABLE:
            missing_features.append("XGBoost")
        if not TRANSFORMERS_AVAILABLE:
            missing_features.append("BERT")
        if not SHAP_AVAILABLE:
            missing_features.append("SHAP/LIME")
        
        if missing_features:
            st.sidebar.warning(f"Install: {', '.join(missing_features)}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem; background: rgba(168, 237, 234, 0.1); border-radius: 12px; margin-bottom: 1rem;">
        <p style="color: #a8edea; font-size: 0.9rem; margin: 0;">💬 <strong>Chatbot available</strong></p>
        <p style="color: rgba(232, 232, 232, 0.7); font-size: 0.8rem; margin: 0.3rem 0 0 0;">Use the navigation menu to access the chatbot page</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.85rem; padding: 1rem;">
        Built with ❤️ using Streamlit<br>
        Explainable AI Pipeline
    </div>
    """, unsafe_allow_html=True)
    
    # Main input section
    st.markdown("### 📝 Enter Text for Analysis")
    user_text = st.text_area(
        "",
        height=180,
        placeholder="Example: I've been struggling with substance use and need help...",
        help="Enter the text you want to analyze for substance-abuse risk"
    )
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        # Add inline style to ensure button text is visible
        st.markdown("""
        <style>
            /* Force all button text to be dark and visible */
            button[kind="primary"],
            button[kind="primary"] > div,
            button[kind="primary"] > div > p,
            button[kind="primary"] span {
                color: #1a1625 !important;
                -webkit-text-fill-color: #1a1625 !important;
                font-weight: 700 !important;
            }
            
            /* Ensure Settings button text is visible */
            [data-testid="stSidebar"] h2 {
                color: #000000 !important;
            }
            
            /* Make sure Settings div text is black */
            [data-testid="stSidebar"] div h2 {
                color: #000000 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze Text", type="primary", use_container_width=True)
    
    if analyze_clicked:
        if not user_text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return
        
        # Predictions section
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 2rem; border-radius: 20px; 
                        box-shadow: 0 20px 60px rgba(102, 126, 234, 0.4), 0 0 0 1px rgba(255,255,255,0.1);
                        backdrop-filter: blur(10px);">
                <h2 style="color: white; margin-top: 0; font-size: 1.9rem; font-weight: 700; font-family: 'Poppins', sans-serif;">📊 Prediction</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if model_choice == 'SVM':
                pred, proba = predict_with_svm(user_text, models)
            elif model_choice == 'RandomForest':
                pred, proba = predict_with_rf(user_text, models)
            elif model_choice == 'XGBoost':
                pred, proba = predict_with_xgb(user_text, models)
            elif model_choice == 'BERT':
                pred, proba = predict_with_bert(user_text, models)
            else:  # Stacked Ensemble
                pred, proba = predict_with_stack(user_text, models)
            
            if pred is not None:
                proba_float = float(proba)
                
                # Risk level display with gradient
                if pred == 1:
                    st.markdown(f"""
                    <div class="risk-high">
                        🔴 HIGH RISK DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="risk-low">
                        🟢 LOW RISK
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability metric with better styling
                st.markdown("<br>", unsafe_allow_html=True)
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Confidence", f"{proba_float:.1%}", delta=None)
                with metric_col2:
                    st.metric("Model", model_choice)
                
                # Progress bar
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Risk Probability:**")
                st.progress(proba_float)
                
            else:
                st.error("❌ Prediction failed. Model not available.")
        
        with col2:
            st.markdown("""
            <div style="background: rgba(255, 255, 255, 0.05); padding: 2rem; border-radius: 20px; 
                        border: 1px solid rgba(168, 237, 234, 0.2); 
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
                        backdrop-filter: blur(10px);">
                <h2 style="color: #e0e0e0; margin-top: 0; font-size: 1.9rem; font-weight: 700; font-family: 'Poppins', sans-serif;">🔬 Explanation</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # Show explanation if LIME or SHAP is available
            if (LIME_AVAILABLE or SHAP_AVAILABLE) and model_choice in ['SVM', 'RandomForest', 'XGBoost', 'BERT']:
                try:
                    # Determine model key and type
                    if model_choice == 'SVM':
                        model_key = 'svm'
                        model_type = 'linear'
                        embedder = models[model_key]['embedder']
                    elif model_choice == 'RandomForest':
                        model_key = 'rf'
                        model_type = 'linear'
                        embedder = models[model_key]['embedder']
                    elif model_choice == 'XGBoost':
                        model_key = 'xgb'
                        model_type = 'xgb'
                        embedder = models[model_key]['embedder']
                    elif model_choice == 'BERT':
                        model_key = 'bert'
                        model_type = 'bert'
                        # For BERT, we need to create a wrapper embedder for LIME
                        embedder = None  # BERT doesn't use TF-IDF embedder
                    else:
                        model_key = None
                        model_type = None
                        embedder = None
                    
                    if model_key and model_key in models:
                        # For BERT, use LIME only (SHAP for transformers is complex)
                        if model_choice == 'BERT':
                            explainer = ModelExplainer(
                                models[model_key]['model'],
                                None,  # No TF-IDF embedder for BERT
                                model_type='bert'
                            )
                            # Store tokenizer for BERT predictions in LIME
                            explainer.tokenizer = models[model_key]['tokenizer']
                            explainer.explainer_shap = None  # SHAP not supported for BERT in this implementation
                        else:
                            # Initialize explainer for SVM/RF/XGBoost (SHAP may fail, but LIME will still work)
                            try:
                                explainer = ModelExplainer(
                                    models[model_key]['model'],
                                    models[model_key]['embedder'],
                                    model_type=model_type
                                )
                            except Exception as e:
                                st.warning(f"SHAP initialization failed (this is okay): {str(e)}")
                                # Create a minimal explainer just for LIME
                                explainer = ModelExplainer(
                                    models[model_key]['model'],
                                    models[model_key]['embedder'],
                                    model_type=model_type
                                )
                                explainer.explainer_shap = None  # Disable SHAP, use LIME only
                        
                        # Try SHAP first (if available) - only for non-BERT models
                        if model_choice != 'BERT':
                            embedder = models[model_key]['embedder']
                        if SHAP_AVAILABLE and explainer.explainer_shap is not None and model_choice != 'BERT':
                            try:
                                st.markdown("""
                                <div style="background: rgba(168, 237, 234, 0.1); padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem;
                                            border: 1px solid rgba(168, 237, 234, 0.3); backdrop-filter: blur(10px);">
                                    <h3 style="color: #a8edea; margin: 0; font-size: 1.4rem; font-weight: 700; font-family: 'Poppins', sans-serif;">📈 SHAP Feature Importance</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                X_sample = embedder.transform([user_text])
                                if isinstance(explainer.explainer_shap, shap.KernelExplainer):
                                    shap_values = explainer.explainer_shap.shap_values(X_sample, nsamples=50)
                                else:
                                    shap_values = explainer.explainer_shap.shap_values(X_sample)
                                
                                # Get feature names
                                try:
                                    feature_names = embedder.get_feature_names()
                                except:
                                    feature_names = [f'Feature {i}' for i in range(X_sample.shape[1])]
                                
                                # Get top features by absolute SHAP value
                                if isinstance(shap_values, list):
                                    shap_vals = shap_values[0] if len(shap_values) > 0 else shap_values
                                elif len(shap_values.shape) > 1:
                                    shap_vals = shap_values[0] if shap_values.shape[0] == 1 else shap_values.mean(axis=0)
                                else:
                                    shap_vals = shap_values
                                
                                # Check if SHAP values are meaningful
                                shap_std = np.std(shap_vals)
                                if shap_std < 1e-6:
                                    # Use feature importance as fallback
                                    st.info("ℹ️ Model predictions are very similar. Showing feature importance instead of SHAP values.")
                                    try:
                                        if hasattr(models[model_key]['model'], 'get_score'):
                                            importance = models[model_key]['model'].get_score(importance_type='gain')
                                            if importance:
                                                # Convert to list and sort
                                                feat_imp = [(k, v) for k, v in importance.items()]
                                                feat_imp.sort(key=lambda x: x[1], reverse=True)
                                                
                                                for feat_name, imp_val in feat_imp[:10]:
                                                    # Try to find matching feature index
                                                    try:
                                                        feat_idx = list(feature_names).index(feat_name)
                                                        color = "#a8edea"
                                                        st.markdown(f"""
                                                        <div class="feature-item" style="border-left-color: {color};">
                                                            <strong>{feat_name}</strong>: <span style="color: {color}; font-weight: 600;">{imp_val:.4f}</span>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    except (ValueError, IndexError):
                                                        continue
                                    except Exception as e:
                                        st.warning(f"Could not get feature importance: {e}")
                                else:
                                    # Sort by absolute value
                                    top_indices = np.argsort(np.abs(shap_vals))[-10:][::-1]
                                    
                                    # Display features in styled list
                                    for idx in top_indices:
                                        feat_name = feature_names[idx] if idx < len(feature_names) else f'Feature {idx}'
                                        shap_val = shap_vals[idx]
                                        color = "#e74c3c" if shap_val > 0 else "#3498db"
                                        st.markdown(f"""
                                        <div class="feature-item" style="border-left-color: {color};">
                                            <strong>{feat_name}</strong>: <span style="color: {color}; font-weight: 600;">{shap_val:.4f}</span>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                st.markdown("<br>", unsafe_allow_html=True)
                            except Exception as e:
                                st.warning(f"⚠️ SHAP explanation failed: {str(e)}")
                        
                        # LIME explanation (works for all models)
                        try:
                            lime_exp = explainer.explain_lime(user_text, num_features=10)
                            if lime_exp is not None:
                                st.markdown("""
                                <div style="background: rgba(168, 237, 234, 0.1); padding: 1.2rem; border-radius: 12px; margin-bottom: 1rem;
                                            border: 1px solid rgba(168, 237, 234, 0.3); backdrop-filter: blur(10px);">
                                    <h3 style="color: #a8edea; margin: 0; font-size: 1.4rem; font-weight: 700; font-family: 'Poppins', sans-serif;">🍋 LIME Top Features</h3>
                                </div>
                                """, unsafe_allow_html=True)
                                try:
                                    # Get explanation as list and handle potential type issues
                                    exp_list = lime_exp.as_list()
                                    if exp_list:
                                        # Check if all weights are zero (LIME might not have found differences)
                                        weights = []
                                        for item in exp_list:
                                            if isinstance(item, tuple) and len(item) == 2:
                                                _, w = item
                                                try:
                                                    w_float = float(w.item() if hasattr(w, 'item') else w)
                                                    weights.append(w_float)
                                                except:
                                                    pass
                                        
                                        all_zero = len(weights) > 0 and all(abs(w) < 1e-6 for w in weights)
                                        
                                        if all_zero:
                                            st.warning("⚠️ LIME couldn't detect feature differences because the model returns similar predictions for all inputs. This suggests the model needs retraining.")
                                            
                                            # Show alternative: word frequency analysis
                                            st.markdown("**Alternative: Word Frequency Analysis**")
                                            words = user_text.lower().split()
                                            from collections import Counter
                                            word_freq = Counter(words)
                                            # Filter out common stop words
                                            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
                                            filtered_freq = {word: count for word, count in word_freq.items() if word not in stop_words and len(word) > 2}
                                            top_words = sorted(filtered_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                                            
                                            for word, count in top_words:
                                                st.markdown(f"""
                                                <div class="feature-item" style="border-left-color: #a8edea;">
                                                    <strong>{word}</strong>: appears <span style="color: #a8edea; font-weight: 600;">{count}</span> time(s)
                                                </div>
                                                """, unsafe_allow_html=True)
                                            
                                            st.info("💡 **To get better explanations:** Retrain the model with more diverse data or adjust hyperparameters. Run: `python src/models/train_xgb.py`")
                                        else:
                                            # Show features with meaningful weights, sorted by absolute value
                                            sorted_exp = sorted(exp_list, key=lambda x: abs(float(x[1].item() if hasattr(x[1], 'item') else x[1])), reverse=True)
                                            for item in sorted_exp[:10]:
                                                if isinstance(item, tuple) and len(item) == 2:
                                                    feature, weight = item
                                                    try:
                                                        weight_float = float(weight.item() if hasattr(weight, 'item') else weight)
                                                        color = "#e74c3c" if weight_float > 0 else "#3498db"
                                                        st.markdown(f"""
                                                        <div class="feature-item" style="border-left-color: {color};">
                                                            <strong>{feature}</strong>: <span style="color: {color}; font-weight: 600;">{weight_float:.4f}</span>
                                                        </div>
                                                        """, unsafe_allow_html=True)
                                                    except (ValueError, TypeError):
                                                        st.write(f"- {feature}: {weight}")
                                                else:
                                                    st.write(f"- {item}")
                                    else:
                                        st.info("No LIME features found.")
                                except Exception as e:
                                    st.warning(f"Could not parse LIME explanation: {e}")
                                    # Try to show raw explanation
                                    try:
                                        st.text(str(lime_exp))
                                    except:
                                        pass
                            else:
                                st.info("LIME explanation returned None")
                        except Exception as e:
                            st.info(f"LIME explanation not available: {str(e)}")
                
                except Exception as e:
                    st.info(f"Explanation features require additional setup: {str(e)}")
            elif not LIME_AVAILABLE:
                st.info("LIME explanations require additional dependencies. Install with: `pip install lime`")
        
        # Model comparison (if multiple models available)
        if len(models) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 1.8rem; border-radius: 20px; 
                        box-shadow: 0 12px 40px rgba(79, 172, 254, 0.4), 0 0 0 1px rgba(255,255,255,0.1);
                        backdrop-filter: blur(10px);">
                <h2 style="color: white; margin: 0; font-size: 1.9rem; font-weight: 700; font-family: 'Poppins', sans-serif;">📊 Model Comparison</h2>
            </div>
            """, unsafe_allow_html=True)
            
            comparison_data = []
            model_functions = [
                ('SVM', predict_with_svm, 'svm'),
                ('RandomForest', predict_with_rf, 'rf'),
                ('XGBoost', predict_with_xgb, 'xgb'),
                ('BERT', predict_with_bert, 'bert')
            ]
            
            for name, func, model_key in model_functions:
                # Check if model is available (with special handling for XGBoost)
                if model_key == 'xgb':
                    if XGBOOST_AVAILABLE and model_key in models:
                        p, prob = func(user_text, models)
                        if p is not None:
                            comparison_data.append({
                                'Model': name,
                                'Prediction': '🔴 High Risk' if p == 1 else '🟢 Low Risk',
                                'Probability': float(prob),  # Store as float for gradient
                                'Probability_Display': f"{prob:.2%}"  # Formatted string for display
                            })
                elif model_key in models:
                    p, prob = func(user_text, models)
                    if p is not None:
                        comparison_data.append({
                            'Model': name,
                            'Prediction': '🔴 High Risk' if p == 1 else '🟢 Low Risk',
                            'Probability': float(prob),  # Store as float for gradient
                            'Probability_Display': f"{prob:.2%}"  # Formatted string for display
                        })
            
            if comparison_data:
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Create fancy comparison cards instead of simple table
                st.markdown("""
                <style>
                    .comparison-container {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 1.5rem;
                        margin: 1.5rem 0;
                    }
                    
                    .model-card {
                        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%);
                        border-radius: 20px;
                        padding: 1.5rem;
                        border: 2px solid rgba(168, 237, 234, 0.3);
                        box-shadow: 0 8px 32px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
                        backdrop-filter: blur(10px);
                        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
                        position: relative;
                        overflow: hidden;
                    }
                    
                    .model-card::before {
                        content: '';
                        position: absolute;
                        top: 0;
                        left: 0;
                        right: 0;
                        height: 4px;
                        background: linear-gradient(90deg, #a8edea 0%, #fed6e3 50%, #d299c2 100%);
                    }
                    
                    .model-card:hover {
                        transform: translateY(-5px) scale(1.02);
                        box-shadow: 0 12px 48px rgba(102, 126, 234, 0.5), inset 0 1px 0 rgba(255,255,255,0.2);
                        border-color: rgba(168, 237, 234, 0.6);
                    }
                    
                    .model-header {
                        display: flex;
                        align-items: center;
                        gap: 0.75rem;
                        margin-bottom: 1rem;
                    }
                    
                    .model-icon {
                        font-size: 2rem;
                        filter: drop-shadow(0 2px 4px rgba(168, 237, 234, 0.5));
                    }
                    
                    .model-name {
                        font-family: 'Poppins', sans-serif;
                        font-size: 1.3rem;
                        font-weight: 700;
                        color: #e0e0e0;
                        margin: 0;
                    }
                    
                    .prediction-badge {
                        display: inline-block;
                        padding: 0.5rem 1rem;
                        border-radius: 25px;
                        font-weight: 700;
                        font-size: 0.9rem;
                        margin-bottom: 0.75rem;
                        font-family: 'Poppins', sans-serif;
                        text-transform: uppercase;
                        letter-spacing: 0.05em;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
                    }
                    
                    .prediction-high {
                        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 50%, #c44569 100%);
                        color: white;
                    }
                    
                    .prediction-low {
                        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 50%, #00d4ff 100%);
                        color: white;
                    }
                    
                    .probability-display {
                        font-family: 'Poppins', sans-serif;
                        font-size: 2rem;
                        font-weight: 800;
                        margin: 0.5rem 0;
                        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;
                        background-clip: text;
                        text-align: center;
                    }
                    
                    .probability-label {
                        font-size: 0.85rem;
                        color: rgba(232, 232, 232, 0.7);
                        text-align: center;
                        margin-top: 0.25rem;
                        font-family: 'Inter', sans-serif;
                    }
                    
                    .model-stats {
                        display: flex;
                        justify-content: space-between;
                        margin-top: 1rem;
                        padding-top: 1rem;
                        border-top: 1px solid rgba(168, 237, 234, 0.2);
                    }
                    
                    .stat-item {
                        text-align: center;
                        flex: 1;
                    }
                    
                    .stat-value {
                        font-family: 'Poppins', sans-serif;
                        font-size: 1.1rem;
                        font-weight: 700;
                        color: #a8edea;
                    }
                    
                    .stat-label {
                        font-size: 0.75rem;
                        color: rgba(232, 232, 232, 0.6);
                        margin-top: 0.25rem;
                    }
                </style>
                """, unsafe_allow_html=True)
                
                # Model icons mapping
                model_icons = {
                    'SVM': '⚡',
                    'RandomForest': '🌲',
                    'XGBoost': '🚀',
                    'BERT': '🧠',
                    'Stacked Ensemble': '🎯'
                }
                
                # Create cards for each model
                cols = st.columns(len(comparison_data))
                for idx, (col, data) in enumerate(zip(cols, comparison_data)):
                    with col:
                        model_name = data['Model']
                        prediction = data['Prediction']
                        prob = data['Probability']
                        prob_display = data['Probability_Display']
                        
                        # Determine if high or low risk
                        is_high_risk = '🔴 High Risk' in prediction
                        badge_class = 'prediction-high' if is_high_risk else 'prediction-low'
                        badge_text = 'HIGH RISK' if is_high_risk else 'LOW RISK'
                        
                        # Get icon
                        icon = model_icons.get(model_name, '📊')
                        
                        st.markdown(f"""
                        <div class="model-card">
                            <div class="model-header">
                                <span class="model-icon">{icon}</span>
                                <h3 class="model-name">{model_name}</h3>
                            </div>
                            <div class="prediction-badge {badge_class}">
                                {badge_text}
                            </div>
                            <div class="probability-display">
                                {prob_display}
                            </div>
                            <div class="probability-label">Confidence Level</div>
                            <div class="model-stats">
                                <div class="stat-item">
                                    <div class="stat-value">{'🔴' if is_high_risk else '🟢'}</div>
                                    <div class="stat-label">Status</div>
                                </div>
                                <div class="stat-item">
                                    <div class="stat-value">{prob:.0%}</div>
                                    <div class="stat-label">Probability</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Also show a summary table below for detailed view
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("""
                <div style="text-align: center; color: rgba(232, 232, 232, 0.7); font-size: 0.9rem; margin-top: 1rem;">
                    💡 Hover over cards for detailed view
                </div>
                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

