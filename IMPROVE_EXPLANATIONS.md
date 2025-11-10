# How to Get Good LIME and SHAP Values

## Problem
The current model returns identical predictions (~0.6) for all inputs, which means:
- LIME shows all zeros (no differences to detect)
- SHAP shows limited variation
- Explanations are not meaningful

## Root Cause
The model needs better training with:
1. More diverse training data
2. Better hyperparameters
3. Proper feature engineering

## Solutions

### Option 1: Retrain XGBoost with Better Parameters (Recommended)

```bash
cd "/Users/athulkrishnaboban/Desktop/luma 2"
source venv/bin/activate
python src/models/train_xgb.py
```

The training script has been updated with improved parameters:
- `max_depth`: 8 (was 6) - deeper trees for complex patterns
- `learning_rate`: 0.05 (was 0.1) - slower learning for better convergence
- Added regularization (`reg_alpha`, `reg_lambda`)
- Added `gamma` and `min_child_weight` for better splits

### Option 2: Check Your Training Data

Ensure `data/train.csv` has:
- **Diverse examples**: Both high-risk and low-risk texts
- **Sufficient samples**: At least 100+ examples per class
- **Quality labels**: Accurate risk assessments

### Option 3: Improve Feature Engineering

The current TF-IDF uses:
- `max_features=5000`
- `ngram_range=(1, 2)` (unigrams and bigrams)

You can improve by:
1. Increasing `max_features` to 10000
2. Adding trigrams: `ngram_range=(1, 3)`
3. Using different preprocessing (stemming, lemmatization)

### Option 4: Use Different Models

Try SVM or RandomForest which might work better with your data:
```bash
python src/models/train_baseline.py
```

## What I've Already Improved

1. **LIME Improvements**:
   - Increased samples to 10000 when model variance is low
   - Added cosine distance metric for better text similarity
   - Added word frequency fallback when LIME shows zeros

2. **SHAP Improvements**:
   - Increased KernelExplainer samples to 200
   - Added feature importance fallback when SHAP values are zero
   - Better error handling

3. **UI Improvements**:
   - Shows feature importance when SHAP/LIME fail
   - Displays word frequency analysis as alternative
   - Clear messages explaining why explanations might be limited

## Quick Test

After retraining, test if the model has variation:
```python
import xgboost as xgb
from src.features.embeddings import TFIDFEmbedder

model = xgb.Booster()
model.load_model('trained_models/xgb_model.json')
embedder = TFIDFEmbedder()
embedder.load('trained_models/xgb_tfidf.pkl')

test_texts = [
    'I feel great today',
    'I am very depressed and want to hurt myself'
]

X = embedder.transform(test_texts)
dmatrix = xgb.DMatrix(X)
preds = model.predict(dmatrix)

print(f"Predictions: {preds}")
print(f"Variance: {preds.var()}")
# If variance > 0.01, the model is working better!
```

## Expected Results After Retraining

- **LIME**: Should show non-zero weights for different words
- **SHAP**: Should show varying importance values
- **Predictions**: Should vary based on input text (not all 0.6)

