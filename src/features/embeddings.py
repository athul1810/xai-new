"""
Feature extraction: TF-IDF and BERT embeddings for text classification.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import joblib

# Optional transformers import
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. BERT embeddings will not work.")

class TFIDFEmbedder:
    """TF-IDF vectorizer wrapper."""
    
    def __init__(self, max_features=5000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True
        )
    
    def fit(self, texts):
        """Fit on training texts."""
        self.vectorizer.fit(texts)
        return self
    
    def transform(self, texts):
        """Transform texts to TF-IDF vectors."""
        return self.vectorizer.transform(texts)
    
    def fit_transform(self, texts):
        """Fit and transform."""
        return self.vectorizer.fit_transform(texts)
    
    def save(self, path):
        """Save vectorizer."""
        joblib.dump(self.vectorizer, path)
    
    def load(self, path):
        """Load vectorizer."""
        self.vectorizer = joblib.load(path)
        return self
    
    def get_feature_names(self):
        """Get feature names from vectorizer."""
        try:
            return self.vectorizer.get_feature_names_out()
        except AttributeError:
            # Fallback for older sklearn versions
            return self.vectorizer.get_feature_names()

class BERTEmbedder:
    """BERT-based embeddings using transformers."""
    
    def __init__(self, model_name='distilbert-base-uncased', device=None):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for BERT embeddings. Install with: pip install transformers torch")
        
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def embed(self, texts, batch_size=32, max_length=128):
        """
        Generate BERT embeddings for texts.
        Returns mean-pooled embeddings.
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Mean pooling
                embeddings_batch = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embeddings_batch)
        
        return np.vstack(embeddings)
    
    def embed_single(self, text, max_length=128):
        """Embed a single text."""
        return self.embed([text], batch_size=1, max_length=max_length)[0]

def load_embeddings(embedder_type='tfidf', texts=None, model_path=None, **kwargs):
    """
    Load or create embeddings.
    
    Args:
        embedder_type: 'tfidf' or 'bert'
        texts: List of texts to embed
        model_path: Path to saved model (for TF-IDF)
        **kwargs: Additional arguments for embedder
    """
    if embedder_type == 'tfidf':
        embedder = TFIDFEmbedder(**kwargs)
        if texts is not None:
            if model_path and Path(model_path).exists():
                embedder.load(model_path)
                return embedder.transform(texts)
            else:
                return embedder.fit_transform(texts)
        return embedder
    
    elif embedder_type == 'bert':
        embedder = BERTEmbedder(**kwargs)
        if texts is not None:
            return embedder.embed(texts)
        return embedder
    
    else:
        raise ValueError(f"Unknown embedder_type: {embedder_type}")

