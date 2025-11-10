"""
Fine-tune DistilBERT for substance-abuse-risk detection.
"""

import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import compute_metrics, plot_confusion_matrix, save_metrics

def compute_metrics_fn(eval_pred):
    """Compute metrics for Trainer."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert(X_train, y_train, X_val, y_val, model_name='distilbert-base-uncased',
               output_dir='trained_models/bert', random_state=42):
    """Fine-tune BERT model."""
    print(f"Fine-tuning {model_name}...")
    
    # Set random seeds
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Tokenize data
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    # Create datasets
    train_dataset = Dataset.from_dict({'text': X_train.tolist(), 'label': y_train.tolist()})
    val_dataset = Dataset.from_dict({'text': X_val.tolist(), 'label': y_val.tolist()})
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        seed=random_state
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_fn
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print(f"BERT Evaluation Results: {eval_results}")
    
    # Predictions
    predictions = trainer.predict(val_dataset)
    y_pred = np.argmax(predictions.predictions, axis=1)
    y_proba = torch.softmax(torch.tensor(predictions.predictions), dim=1)[:, 1].numpy()
    
    # Compute metrics
    metrics = compute_metrics(y_val, y_pred, y_proba)
    print(f"BERT Metrics: {metrics}")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'trainer': trainer,
        'metrics': metrics,
        'predictions': y_pred,
        'probabilities': y_proba
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
    Path('trained_models/bert').mkdir(parents=True, exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Train BERT
    bert_results = train_bert(X_train, y_train, X_val, y_val)
    
    # Save model
    bert_results['model'].save_pretrained('trained_models/bert')
    bert_results['tokenizer'].save_pretrained('trained_models/bert')
    
    # Save metrics
    save_metrics(bert_results['metrics'], 'BERT')
    plot_confusion_matrix(y_val, bert_results['predictions'],
                         save_path='outputs/bert_confusion_matrix.png')
    
    print("\nBERT model trained and saved!")

if __name__ == "__main__":
    main()

