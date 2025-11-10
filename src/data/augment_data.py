"""
Data augmentation script to expand the training dataset.
Creates variations of existing examples to improve model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import random

def augment_data(input_path='data/raw.csv', output_path='data/raw_augmented.csv', multiplier=10):
    """
    Augment dataset by creating variations of existing examples.
    
    Args:
        input_path: Path to original CSV
        output_path: Path to save augmented CSV
        multiplier: How many times to expand the dataset
    """
    df = pd.read_csv(input_path)
    
    # High-risk patterns
    high_risk_patterns = [
        "struggling", "addiction", "substance", "drug", "alcohol", "drinking", 
        "using", "dose", "cope", "stress", "anxiety", "depression", "pain",
        "hurt", "suffer", "torture", "cry", "overwhelmed", "hopeless", "suicide"
    ]
    
    # Low-risk patterns
    low_risk_patterns = [
        "great", "happy", "wonderful", "productive", "energized", "healthy",
        "family", "friends", "workout", "exercise", "positive", "grateful"
    ]
    
    augmented_data = []
    
    for _, row in df.iterrows():
        text = str(row['text']).lower()
        label = row['label']
        
        # Add original
        augmented_data.append({'text': row['text'], 'label': label})
        
        # Create variations
        for i in range(multiplier - 1):
            # Variation 1: Add context words
            if label == 1:
                # Add high-risk context
                context_words = random.sample(high_risk_patterns, k=random.randint(1, 3))
                new_text = f"{' '.join(context_words)} {text}"
            else:
                # Add low-risk context
                context_words = random.sample(low_risk_patterns, k=random.randint(1, 3))
                new_text = f"{' '.join(context_words)} {text}"
            
            augmented_data.append({'text': new_text, 'label': label})
    
    # Add more diverse examples
    additional_examples = [
        # High risk
        ("I can't stop drinking every single day", 1),
        ("Using drugs to escape reality and pain", 1),
        ("My addiction is controlling my entire life", 1),
        ("Need help with substance abuse problem", 1),
        ("Drinking heavily to numb emotional pain", 1),
        ("Can't function without my medication", 1),
        ("Struggling with alcohol dependency", 1),
        ("Drug use has ruined my relationships", 1),
        ("Feeling hopeless about my addiction", 1),
        ("Using substances to cope with trauma", 1),
        
        # Low risk
        ("Feeling grateful and positive today", 0),
        ("Had an amazing day with loved ones", 0),
        ("Feeling healthy and energized", 0),
        ("Enjoying life and staying positive", 0),
        ("Feeling motivated and productive", 0),
        ("Having a great time with friends", 0),
        ("Feeling happy and content", 0),
        ("Maintaining healthy habits daily", 0),
        ("Feeling optimistic about the future", 0),
        ("Enjoying peaceful and calm moments", 0),
    ]
    
    for text, label in additional_examples:
        augmented_data.append({'text': text, 'label': label})
    
    # Shuffle
    random.shuffle(augmented_data)
    
    # Create DataFrame
    augmented_df = pd.DataFrame(augmented_data)
    
    # Save
    augmented_df.to_csv(output_path, index=False)
    print(f"Created augmented dataset: {output_path}")
    print(f"Original samples: {len(df)}")
    print(f"Augmented samples: {len(augmented_df)}")
    print(f"Label distribution:")
    print(augmented_df['label'].value_counts())
    
    return augmented_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment training data')
    parser.add_argument('--input', type=str, default='data/raw.csv', help='Input CSV file')
    parser.add_argument('--output', type=str, default='data/raw_augmented.csv', help='Output CSV file')
    parser.add_argument('--multiplier', type=int, default=10, help='Dataset expansion multiplier')
    
    args = parser.parse_args()
    
    augment_data(args.input, args.output, args.multiplier)

