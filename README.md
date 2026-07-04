# Sentiment Analysis: Traditional ML vs LLMs
A research project comparing traditional Machine Learning pipelines against Large Language Models for sentiment classification on the IMDb Movie Reviews dataset.

## Project Overview
- **Dataset:** IMDb Movie Reviews (50,000 reviews)
- **Task:** Binary sentiment classification (Positive/Negative)
- **Goal:** Evaluate whether traditional ML can outperform zero-shot LLMs

## Methods Implemented

### Vectorization
- Count Vectorizer (Bag of Words)
- TF-IDF
- GloVe Embeddings

### Dimensionality Reduction
- Chi-Square Feature Selection
- Truncated SVD
- PCA

### ML Models
- Logistic Regression (L1 and L2)
- Linear SVM
- Naive Bayes
- LDA and QDA
- SVM with RBF Kernel

### LLM Baseline
- Llama 3.1 8B Zero-shot via Groq API

## Results

| Method | F1 Score | Inference Time |
|--------|----------|----------------|
| TF-IDF + Logistic L2 | 0.8797 | ~0.1ms |
| TF-IDF + Logistic L1 | 0.8787 | ~0.1ms |
| SVD + Logistic L2 | 0.8783 | ~0.1ms |
| Llama 3.1 8B zero-shot | 0.8744 | ~1994ms |
| PCA GloVe + SVM RBF | 0.8015 | ~5ms |

## Key Findings
- Traditional ML outperformed zero-shot LLM in accuracy
- Traditional ML is 20,000x faster than LLM inference
- Dimensionality reduction retained 99% performance with 80% fewer features
- TF-IDF outperformed GloVe regardless of classifier used

## Project Structure
- data.ipynb — Data loading and cleaning
- vectorization.ipynb — Feature extraction (Count Vectorizer, TF-IDF, GloVe)
- dimensionality_reduction.ipynb — Dimensionality reduction and visualizations
- ml_vs_llm.ipynb — ML models training and LLM comparison

## Dependencies
Run this to install all required libraries:

pip install numpy pandas scikit-learn matplotlib seaborn nltk gensim jupyter datasets groq

