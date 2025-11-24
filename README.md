# modernbert-IDS
Intrusion Detection System based on ModerBERT model

# ModernBERT-DoS: Fine-Tuned Transformer for Network Traffic Log Classification

This repository contains a complete training and evaluation pipeline for **ModernBERT-DoS**, a ModernBERT-based multi-class log classification model designed to detect DDoS and related attack categories from raw network logs.

The notebook implements:

- End-to-end dataset ingestion  
- Data leakage–safe splitting  
- ModernBERT fine-tuning with automatic feature learning  
- Early stopping with F1-based training halt  
- Full evaluation on unseen test data  
- External dataset generalization  
- Traditional ML baseline comparisons (RF, SVM, LR)  
- Deep learning baselines (CNN, LSTM-Attention)  
- Comprehensive visualization and confusion matrix  
- Final performance ranking table  

---

## Key Features

### 1. Raw Log Processing
- The model learns directly from raw log strings.  
- No need to manually remove IPs, ports, timestamps, etc.  
- ModernBERT automatically filters non-useful features.

### 2. Strict Data Leakage Prevention
Two-stage split:
- Train/validation vs. test  
- Train vs. validation  

Ensures the test set is never exposed during training or TF-IDF fitting.

### 3. ModernBERT Fine-Tuning
- Model: `answerdotai/ModernBERT-base`  
- Architecture:  
  - Mean pooling  
  - 3× dropout  
  - 2 fully-connected feature layers  
  - LayerNorm + GELU  
  - Weighted cross-entropy for unbalanced classes  
- Tokenization at `max_length=512` to support large logs.

### 4. Advanced Training Controls
- EarlyStopping (patience=5)  
- Custom F1ThresholdCallback (stop when macro-F1 ≥ 0.95)  
- Gradient accumulation  
- FP16 support  
- Best-model checkpointing  

### 5. Traditional ML Baselines
Trained without leakage, using TF-IDF:
- Random Forest  
- Linear SVM  
- Logistic Regression  

### 6. Deep Learning Baselines
Vocabulary built only on training data:
- Enhanced CNN with multiple kernel sizes  
- Two-layer BiLSTM with attention  

### 7. Generalization Testing
Supports uploading an external dataset to test model performance in unseen environments.

### 8. Full Evaluation Suite
Includes:
- Accuracy  
- Macro and Weighted F1  
- Precision and Recall  
- Confusion matrix  
- Classification report  
- Training curve visualization  
- Final model ranking chart  

---

## Notebook Workflow

### 1. Upload Training Dataset
CSV containing:
- Log (raw text)  
- Attack Type (label)

### 2. Build Label Mappings
Automatically builds class → ID dictionary.

### 3. Train/Validation/Test Split
Stratified splits with no leakage.

### 4. Tokenization
AutoTokenizer (ModernBERT), 512 tokens.

### 5. Model Definition
Custom classifier on top of ModernBERT.

### 6. Training (HuggingFace Trainer)
Metrics tracked:
- Accuracy  
- Macro F1  
- Weighted F1  
- Per-class F1  

### 7. Test Evaluation
Evaluates only on unseen logs.

### 8. Baseline Comparisons
RF, SVM, LR, CNN, LSTM.

### 9. External Dataset Evaluation
Upload a new CSV and run full ModernBERT evaluation.

### 10. Charts and Visualizations
- Confusion matrix  
- Accuracy/F1 bar charts  
- Training loss and metric curves  

---

## Outputs
The notebook automatically generates:
- Best checkpoint (`./modernbert_pcaps_logs/`)  
- Model comparison tables  
- Heatmaps and plots  
- Classification report  
- Generalization metrics  

---

## Requirements
transformers
datasets
pandas
numpy
torch
scikit-learn
matplotlib
seaborn

---

## Supported Attack Classes
Detected dynamically from the dataset, typically including:
- DDoS  
- BENIGN  
- LDAP  
- NetBIOS  
- MSSQL  
- Portmap  
- UDP  
- SSL  
- Others present in the CSV  

---

## Final Deliverable
The notebook returns:
- Final ModernBERT Accuracy  
- Final ModernBERT Macro F1  
- 95%+ Macro-F1 target validation  
- Full comparison with baselines  
- Generalization results on external logs  

---

## Usage
1. Open the notebook in Google Colab.  
2. Upload your dataset (`training_dataset.csv`).  
3. Run all cells.  
4. Optionally upload an external dataset for generalization testing.  

---

## Model Availability

The fine-tuned ModernBERT-DoS model is publicly available on Hugging Face:

**https://huggingface.co/ccaug/modernbert-IDS**

You can download, load, and run inference directly using the Hugging Face Transformers library.
