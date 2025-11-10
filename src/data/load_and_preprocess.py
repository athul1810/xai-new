"""
Data loading and preprocessing pipeline for substance-abuse-risk detection.
Handles text cleaning, lemmatization, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
import re
import spacy
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load SpaCy model (run: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Warning: SpaCy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = None

def clean_text(text):
    """
    Clean text: lowercase, remove URLs, mentions, emojis, extra whitespace.
    """
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags (keep the word)
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove emojis
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove punctuation except basic sentence punctuation
    text = re.sub(r'[^\w\s\.\!\?]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def lemmatize_text(text, nlp_model=None):
    """
    Lemmatize text using SpaCy.
    """
    if nlp_model is None:
        return text
    
    doc = nlp_model(text)
    lemmatized = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(lemmatized)

def preprocess_dataframe(df, text_col='text', label_col='label', lemmatize=True):
    """
    Preprocess entire dataframe.
    """
    df = df.copy()
    
    # Clean text
    print("Cleaning text...")
    df[text_col] = df[text_col].apply(clean_text)
    
    # Lemmatize if requested and model available
    if lemmatize and nlp is not None:
        print("Lemmatizing text...")
        df[text_col] = df[text_col].apply(lambda x: lemmatize_text(x, nlp))
    
    # Remove empty texts
    df = df[df[text_col].str.len() > 0].reset_index(drop=True)
    
    return df

def load_and_preprocess(input_path='data/raw.csv', output_dir='data', 
                       test_size=0.2, val_size=0.1, random_state=42):
    """
    Main preprocessing pipeline: load, clean, split, and save.
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"Loading data from {input_path}...")
    if not Path(input_path).exists():
        print(f"Warning: {input_path} not found. Creating sample data...")
        # Create sample data for testing
        sample_data = {
            'text': [
                "I've been drinking heavily every night and can't stop.",
                "Feeling great today! Had a productive morning.",
                "Struggling with substance use and need help.",
                "Just finished a workout, feeling energized!",
                "Can't function without my daily dose.",
                "Had a wonderful day with family and friends.",
                "Using drugs to cope with stress and anxiety.",
                "Enjoying a healthy lifestyle and good habits."
            ],
            'label': [1, 0, 1, 0, 1, 0, 1, 0]
        }
        df = pd.DataFrame(sample_data)
        df.to_csv(input_path, index=False)
        print(f"Created sample data at {input_path}")
    
    df = pd.read_csv(input_path)
    
    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Data must contain 'text' and 'label' columns")
    
    # Preprocess
    df_clean = preprocess_dataframe(df)
    
    # Split data: train -> val/test -> val/test
    print("Splitting data...")
    X = df_clean[['text']]
    y = df_clean['label']
    
    # Check if we have enough samples for stratified split
    min_samples_per_class = min(y.value_counts().values)
    use_stratify = min_samples_per_class >= 2
    
    # First split: train vs (val + test)
    if use_stratify:
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state, stratify=y
        )
    else:
        print("Warning: Not enough samples for stratified split. Using random split.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=random_state
        )
    
    # Second split: val vs test
    val_size_adjusted = val_size / (test_size + val_size)
    if use_stratify and len(y_temp) > 1:
        min_temp = min(y_temp.value_counts().values) if len(y_temp.value_counts()) > 1 else 0
        use_stratify_temp = min_temp >= 2
    else:
        use_stratify_temp = False
    
    if use_stratify_temp:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size_adjusted), random_state=random_state, stratify=y_temp
        )
    else:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size_adjusted), random_state=random_state
        )
    
    # Combine back with labels
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save splits
    train_path = Path(output_dir) / 'train.csv'
    val_path = Path(output_dir) / 'val.csv'
    test_path = Path(output_dir) / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Saved train set: {train_path} ({len(train_df)} samples)")
    print(f"Saved val set: {val_path} ({len(val_df)} samples)")
    print(f"Saved test set: {test_path} ({len(test_df)} samples)")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess data for substance-abuse-risk detection')
    parser.add_argument('--input', type=str, default='data/raw.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='data', help='Output directory')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    load_and_preprocess(
        input_path=args.input,
        output_dir=args.output,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed
    )

