# Twitter Sentiment Analysis: Traditional ML & RoBERTa Approaches

## Overview

This project provides a comprehensive pipeline for sentiment analysis on a large-scale Twitter dataset (1.6M tweets), featuring both traditional machine learning models and a modern RoBERTa transformer-based approach. The workflows are modular, reproducible, and scalable, supporting both CPU and GPU environments.

---

## Approaches

### 1. Traditional Machine Learning
- **Workflow:**
  - Data loading, cleaning, and preprocessing (NLTK-based)
  - Exploratory Data Analysis (EDA): distributions, word clouds, n-grams
  - Feature engineering: TF-IDF, tweet length, etc.
  - Model training: Naive Bayes, Logistic Regression, Linear SVM
  - Evaluation: metrics, confusion matrices, ROC curves
  - Model interpretability and export
- **Outputs:**
  - Cleaned CSVs, EDA plots, trained models (`.joblib`)

### 2. RoBERTa Transformer Model
- **Workflow:**
  - Minimal preprocessing (preserves information for transformer)
  - Stratified sampling for balanced classes
  - Efficient batching and DataLoader setup
  - Training, validation, and test splits
  - RoBERTa fine-tuning with progress monitoring
  - Evaluation: accuracy, confusion matrix, classification report
  - Model and tokenizer export
- **Outputs:**
  - Best model and tokenizer (`./roberta_twitter_model`), training history plots, confusion matrix

---

## Dataset
- **Source:** `training.1600000.processed.noemoticon.csv` ([Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140))
- **Columns:**
  - `target`: Sentiment (0 = Negative, 4 = Positive)
  - `ids`, `date`, `flag`, `user`, `text`: Metadata and tweet content
- **Class Distribution:** Highly imbalanced (positive/negative only)

---

## Requirements

- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- nltk (traditional ML only)
- wordcloud (traditional ML only)
- scikit-learn
- joblib (traditional ML only)
- emoji (traditional ML only)
- torch (PyTorch, RoBERTa only)
- transformers (RoBERTa only)
- tqdm (RoBERTa only)
- psutil (RoBERTa only)

**Install all requirements:**
```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn joblib emoji torch transformers tqdm psutil
```

---

## Usage

1. **Download the dataset** and place it in the project directory.
2. **Choose your approach:**
   - For traditional ML, run the corresponding notebook/script for data cleaning, EDA, feature engineering, and model training.
   - For RoBERTa, run the transformer notebook/script for minimal preprocessing, tokenization, and model fine-tuning.
3. **Outputs** (plots, processed data, models) will be saved in the project directory.
4. **Prediction:** Use the provided functions to classify new tweets with the best model.

---

## Workflow Details

### Traditional ML Pipeline
- **Data Loading:** Reads CSV, displays info, checks class distribution
- **Preprocessing:** Cleans tweets (URLs, mentions, hashtags, emojis, etc.), tokenizes, removes stopwords, lemmatizes
- **EDA:** Tweet length analysis, word clouds, n-gram frequency, distinctive terms
- **Feature Engineering:** TF-IDF, tweet length, word count
- **Modeling:** Trains/evaluates Naive Bayes, Logistic Regression, SVM; compares metrics
- **Interpretability:** Feature importance for interpretable models
- **Saving:** Cleaned data, plots, best model

### RoBERTa Pipeline
- **Hardware Check:** Detects GPU/CPU, prints resources
- **Data Loading:** Reads CSV, removes missing/empty rows, samples balanced classes
- **Preprocessing:** Minimal (URLs, mentions, whitespace)
- **Splitting:** Stratified train/val/test
- **Tokenization:** RoBERTa tokenizer (max length 48)
- **DataLoader:** Efficient batching
- **Training:** AdamW optimizer, linear warmup, gradient clipping
- **Evaluation:** Accuracy, loss, classification report, confusion matrix
- **Saving:** Best model/tokenizer, training history, confusion matrix
- **Prediction:** Function for new tweets

---

## Outputs
- Cleaned CSVs: `processed_twitter_sentiment_full.csv`, `processed_twitter_sentiment_clean.csv`
- Plots: Sentiment distribution, tweet length, word clouds, n-gram charts, confusion matrices, ROC curves
- Models: `best_model_<model_name>.joblib` (traditional), `./roberta_twitter_model` (transformer)

---

## Troubleshooting & Tips
- **Large Dataset:** Processing 1.6M tweets may require significant memory/time. Use samples for prototyping.
- **GPU Recommended:** RoBERTa training is much faster on GPU.
- **NLTK Data:** Downloaded automatically for traditional ML; ensure internet access.
- **Missing Packages:** Install all requirements before running.
- **Custom Paths:** Update file paths as needed.
- **Reproducibility:** Random seeds are set where possible.
- **Memory Issues:** Reduce sample/batch size if needed.

---

## References & Further Reading
- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [WordCloud Documentation](https://github.com/amueller/word_cloud)
- [Text Classification with scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/index)

---

## Notes
- The code is modular and can be adapted for other text classification tasks.
- For best results, use a machine with sufficient RAM and CPU/GPU resources.
- Respect dataset and model licenses for research/educational use.
