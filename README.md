# Twitter Sentiment Analysis Project

## Overview

This project performs comprehensive sentiment analysis on a large-scale Twitter dataset using Python. The workflow covers data loading, preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning modeling to classify tweets as positive or negative. The notebook is designed to be modular, reproducible, and scalable for large datasets.

## Dataset

- **Source:** `training.1600000.processed.noemoticon.csv` (1.6 million tweets)
- **Columns:**
  - `target`: Sentiment label (0 = Negative, 4 = Positive)
  - `ids`, `date`, `flag`, `user`, `text`: Metadata and tweet content
- **Class Distribution:**
  - The dataset is highly imbalanced, with only positive and negative classes (neutral is rare or absent).

## Project Workflow

### 1. Data Loading

- Reads the CSV file and displays dataset info, shape, and a sample of tweets.
- Checks sentiment class distribution and maps sentiment values to human-readable labels.
- Visualizes sentiment distribution and prints example tweets for each class.

### 2. Data Preprocessing & Cleaning

- Cleans tweets by removing URLs, mentions, hashtags, emojis, emoticons, slang, punctuation, and numbers.
- Handles contractions and normalizes text (lowercasing, removing HTML, etc.).
- Uses NLTK for tokenization, stopword removal (with negation handling), and lemmatization.
- Processes a sample for verification, then the full dataset.
- Saves the processed dataset and analyzes missing values, visualizing missing data and removing problematic rows.
- Outputs: `processed_twitter_sentiment_full.csv`, `processed_twitter_sentiment_clean.csv`

### 3. Exploratory Data Analysis (EDA)

- Analyzes tweet lengths (character and word count), provides summary statistics, and visualizes distributions by sentiment.
- Generates word clouds for all, positive, and negative tweets.
- Computes and visualizes the most common unigrams, bigrams, and trigrams for each sentiment.
- Identifies and prints the most distinctive terms for positive and negative tweets using n-gram frequency ratios.
- Provides recommendations for deep learning sequence length based on tweet word counts.
- Outputs: `sentiment_distribution.png`, `tweet_length_analysis_full.png`, `wordcloud_analysis_full.png`, `top_Uni/Bi/Trigrams_full.png`

### 4. Feature Engineering

- Extracts features such as character length, word count, and TF-IDF vectors (unigrams, bigrams, trigrams).
- Prepares data for machine learning models.

### 5. Machine Learning Modeling & Evaluation

- Converts sentiment labels to numeric values.
- Splits the data into training and test sets (stratified).
- Trains and evaluates Naive Bayes, Logistic Regression, and Linear SVM classifiers using TF-IDF features.
- Compares model performance, prints metrics, and identifies the best model.
- Displays feature importance for interpretable models.
- Plots confusion matrices and ROC curves for all models.
- Saves the best-performing model for future use.
- Outputs: `ml_confusion_matrices.png`, `roc_curves.png`, `best_model_<model_name>.joblib`

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- nltk
- wordcloud
- scikit-learn
- joblib
- emoji

Install requirements with:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud scikit-learn joblib emoji
```

## Usage

1. Place the dataset at the specified path in the notebook.
2. Run the notebook cells in order for end-to-end analysis.
3. Outputs include visualizations, processed datasets, and trained models.

## Outputs

- Cleaned CSV files: `processed_twitter_sentiment_full.csv`, `processed_twitter_sentiment_clean.csv`
- Plots: Sentiment distribution, tweet length analysis, word clouds, n-gram bar charts, confusion matrices, ROC curves
- Best model: Saved as `best_model_<model_name>.joblib`

## Troubleshooting & Tips

- **Large Dataset:** Processing 1.6M tweets may require significant memory and time. Consider working with a sample for rapid prototyping.
- **NLTK Data:** The notebook automatically downloads required NLTK resources. If you encounter errors, ensure you have internet access and write permissions to the NLTK data directory.
- **Missing Packages:** Install all required packages before running the notebook.
- **Reproducibility:** Set random seeds where possible for reproducible results.
- **Custom Paths:** Update file paths as needed for your environment.

## References & Further Reading

- [Sentiment Analysis on Twitter Data](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [NLTK Documentation](https://www.nltk.org/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [WordCloud Documentation](https://github.com/amueller/word_cloud)
- [Text Classification with scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html)

## Notes

- The notebook is designed for large datasets and may take time to run all cells.
- Ensure all required Python packages are installed before running.
- The code is modular and can be adapted for other text classification tasks.
- For best results, use a machine with sufficient RAM and CPU resources.
