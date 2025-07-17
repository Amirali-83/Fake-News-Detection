# Fake News Detection Using RoBERTa and Gradio

## 📖 Project Overview

This project implements a fake news detection system using the RoBERTa transformer model, fine-tuned on a balanced dataset. It combines both headlines and article content for improved accuracy. The system is deployed via a Gradio interface for real-time predictions with confidence scores.

## Features

- RoBERTa-based binary classification (FAKE vs. REAL)
- Title + content merged for richer context
- Balanced dataset (real/fake news from Kaggle)
- Gradio web interface with prediction confidence display

## Model Performance

- **Validation Accuracy:** 99.99%
- **Precision, Recall, F1-Score:** 100% for both classes
- **Confusion Matrix:** 1 error in 8,567 predictions

## Technologies

- Python
- PyTorch
- HuggingFace Transformers
- Gradio
- Pandas, Scikit-learn
