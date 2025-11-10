# Explainable AI Pipeline for Substance-Abuse-Risk Detection

A comprehensive machine learning pipeline for detecting substance-abuse risk in text data, featuring multiple models (SVM, XGBoost, BERT, Stacked Ensemble) with SHAP and LIME explainability, integrated with an optional Ollama conversational explanation layer.

## 🎯 Features

- **Multiple ML Models**: SVM, RandomForest, XGBoost, DistilBERT, and Stacked Ensemble
- **Explainability**: SHAP (global & local) and LIME explanations
- **Web Interface**: Streamlit app with interactive predictions and visualizations
- **Ollama Integration**: Optional natural language explanations via Ollama LLM
- **Reproducible**: Seed-based randomization and modular code structure
- **Comprehensive Evaluation**: Metrics, confusion matrices, and model comparison

## 📁 Repository Structure

```
.
├── src/
│   ├── data/
│   │   └── load_and_preprocess.py    # Data preprocessing pipeline
│   ├── features/
│   │   └── embeddings.py             # TF-IDF and BERT embeddings
│   ├── models/
│   │   ├── train_baseline.py         # SVM & RandomForest
│   │   ├── train_xgb.py              # XGBoost training
│   │   ├── train_bert.py             # BERT fine-tuning
│   │   └── stack.py                  # Ensemble stacking
│   ├── explain/
│   │   └── shap_lime.py             # SHAP & LIME explainability
│   ├── utils/
│   │   └── metrics.py                # Evaluation metrics
│   └── app/
│       └── streamlit_app.py          # Web interface
├── notebooks/
│   ├── exploration.ipynb            # Data exploration
│   └── model_compare.ipynb          # Model comparison
├── trained_models/                  # Saved models
├── outputs/                         # Results and visualizations
│   ├── shap_plots/
│   ├── lime_plots/
│   └── metrics.csv
├── data/                            # Dataset
│   ├── raw.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── requirements.txt
```

## 🚀 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download SpaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 3. (Optional) Setup Ollama

For natural language explanations:

```bash
# Install Ollama
brew install ollama  # macOS
# or download from https://ollama.ai

# Pull model
ollama pull llama3

# Start Ollama server (in separate terminal)
ollama serve
```

## 📊 Data Preparation

### Prepare Your Data

Create `data/raw.csv` with columns:
- `text`: Input text (string)
- `label`: Binary label (0 = Low Risk, 1 = High Risk)

Example:
```csv
text,label
"I've been drinking heavily every night and can't stop.",1
"Feeling great today! Had a productive morning.",0
```

### Preprocess Data

```bash
python src/data/load_and_preprocess.py
```

This will:
- Clean and lemmatize text
- Split into train/val/test sets
- Save processed data to `data/` directory

## 🏋️ Model Training

### 1. Baseline Models (SVM & RandomForest)

```bash
python src/models/train_baseline.py
```

### 2. XGBoost

```bash
python src/models/train_xgb.py
```

### 3. BERT Fine-tuning

```bash
python src/models/train_bert.py
```

**Note**: BERT training requires GPU for reasonable speed. CPU training is possible but slow.

### 4. Stacked Ensemble

Train after XGBoost and BERT are trained:

```bash
python src/models/stack.py
```

## 🔍 Explainability

### Generate SHAP Explanations

```bash
python src/explain/shap_lime.py
```

This creates:
- `outputs/shap_plots/shap_summary.png`: Global feature importance
- `outputs/lime_plots/lime_explanation_*.html`: Local explanations

## 📈 Evaluation

Metrics are automatically saved to `outputs/metrics.csv` during training.

View comparison:

```python
from src.utils.metrics import compare_models
compare_models()
```

Or use the notebook:

```bash
jupyter notebook notebooks/model_compare.ipynb
```

## 🌐 Web Application

### Run Streamlit App

```bash
streamlit run src/app/streamlit_app.py
```

The app provides:
- **Text Input**: Enter text for risk detection
- **Model Selection**: Choose between XGBoost, BERT, or Stacked Ensemble
- **Predictions**: Risk level and probability
- **SHAP Visualizations**: Feature importance plots
- **LIME Explanations**: Top influential tokens
- **Ollama Chat**: Natural language explanations (if enabled)

### Features

1. **Prediction**: Real-time risk assessment
2. **Visualizations**: SHAP summary plots and LIME feature weights
3. **Model Comparison**: Side-by-side predictions from multiple models
4. **Ollama Integration**: Conversational explanations of model decisions

## 📓 Notebooks

### Data Exploration

```bash
jupyter notebook notebooks/exploration.ipynb
```

Explores:
- Label distribution
- Text length statistics
- Word frequency analysis

### Model Comparison

```bash
jupyter notebook notebooks/model_compare.ipynb
```

Compares:
- Model metrics (accuracy, F1, ROC-AUC)
- SHAP visualizations
- Confusion matrices

## 📊 Output Files

- `outputs/metrics.csv`: Model performance metrics
- `outputs/*_confusion_matrix.png`: Confusion matrices
- `outputs/shap_plots/`: SHAP visualizations
- `outputs/lime_plots/`: LIME HTML explanations
- `trained_models/`: Saved model files

## 🔧 Configuration

### Model Parameters

Edit training scripts to adjust:
- **XGBoost**: `max_depth`, `learning_rate`, `n_estimators`
- **BERT**: `num_train_epochs`, `batch_size`, `learning_rate`
- **TF-IDF**: `max_features`, `ngram_range`

### Ollama Settings

In Streamlit app sidebar:
- Enable/disable Ollama
- Select model (llama3, llama2, mistral)

## 🧪 Testing

Run preprocessing on sample data:

```bash
python src/data/load_and_preprocess.py
```

The script will create sample data if `data/raw.csv` doesn't exist.

## 📝 Example Usage

### Command Line

```python
# Load and preprocess
from src.data.load_and_preprocess import load_and_preprocess
train_df, val_df, test_df = load_and_preprocess()

# Train XGBoost
from src.models.train_xgb import train_xgboost
results = train_xgboost(X_train, y_train, X_val, y_val)

# Generate explanations
from src.explain.shap_lime import generate_shap_explanations
generate_shap_explanations('trained_models/xgb_model.json', 
                           'trained_models/xgb_tfidf.pkl',
                           sample_texts)
```

### Streamlit App

1. Start app: `streamlit run src/app/streamlit_app.py`
2. Enter text in input box
3. Select model
4. Click "Analyze"
5. View predictions, SHAP plots, and LIME explanations
6. (Optional) Enable Ollama for natural language explanations

## 🐛 Troubleshooting

### Ollama Connection Error

If Ollama explanations fail:
- Ensure Ollama is running: `ollama serve`
- Check model is available: `ollama list`
- Verify API endpoint: `curl http://localhost:11434/api/generate`

### Model Not Found

If models aren't loading:
- Ensure models are trained first
- Check file paths in `trained_models/`
- Verify model files exist

### GPU Issues (BERT)

If BERT training is slow:
- Install CUDA-enabled PyTorch for GPU acceleration
- Reduce `batch_size` for limited memory
- Use CPU (slower but works)

## 📚 Dependencies

See `requirements.txt` for full list. Key dependencies:
- scikit-learn, xgboost, lightgbm
- transformers, torch
- shap, lime
- streamlit, pandas, numpy
- spacy

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## 📄 License

MIT License - see LICENSE file

## 🙏 Acknowledgments

- Hugging Face Transformers
- SHAP and LIME libraries
- Ollama for LLM integration
- Streamlit for web interface

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is a research/educational tool. For production use, ensure proper validation, testing, and compliance with relevant regulations.
