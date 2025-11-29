The app.py script implements a complete sentiment analysis pipeline that:

📊 What It Does
Data Loading & Exploration

Loads 100,000 sampled reviews (configurable to full 1.44M)
Shows sentiment distribution: ~67% Positive, ~33% Negative
Text Preprocessing

Cleans reviews (lowercase, remove URLs, HTML, punctuation)
Removes missing values
Creates binary labels (1=Positive, 0=Negative)
Trains 4 Classifiers

Model	Accuracy	F1-Score
Logistic Regression	79.06%	85.35% ⭐
Naive Bayes	77.19%	84.60%
Linear SVM	78.30%	84.38%
Random Forest	75.36%	82.80%
Feature Importance Analysis

Positive words: entertaining, enjoyable, perfect, powerful, masterpiece
Negative words: fails, unfortunately, dull, worst, boring
Outputs Generated

sentiment_analysis_eda.png - Exploratory visualizations
model_comparison.png - Model performance comparison
feature_importance.png - Top words for each sentiment
best_model.joblib - Saved best model for future predictions
🎯 Best Model: Logistic Regression with 85.35% F1-Score
The script also includes an interactive demo showing predictions on sample reviews with confidence scores!

