# 🎬 Movie Review Sentiment Analysis - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Imports and Dependencies](#imports-and-dependencies)
3. [Pipeline Architecture](#pipeline-architecture)
4. [Detailed Function Breakdown](#detailed-function-breakdown)
5. [Execution Modes](#execution-modes)
6. [Output Files](#output-files)
7. [Model Performance](#model-performance)
8. [How to Use](#how-to-use)

---

## Project Overview

This project implements a **Sentiment Analysis System** for movie reviews. It trains machine learning classifiers to predict whether a movie review expresses a **POSITIVE** or **NEGATIVE** sentiment based on the review text.

### Key Features:
- Loads and analyzes 1.44 million movie reviews (samples 100,000 for faster processing)
- Trains and compares 4 different machine learning models
- Generates visualizations for exploratory data analysis
- Provides feature importance analysis (which words indicate positive/negative sentiment)
- Includes an interactive web UI for real-time predictions
- Saves the best model for future use

### Dataset Information:
- **Source:** Movie critic reviews from various publications
- **Size:** 1.44 million reviews
- **Target Variable:** `scoreSentiment` (POSITIVE or NEGATIVE)
- **Main Feature:** `reviewText` (the full text of the review)

---

## Imports and Dependencies

### Data Manipulation & Analysis

```python
import pandas as pd
```
- **Purpose:** Data manipulation and analysis library
- **Used for:** Loading CSV data, handling DataFrames, data cleaning, filtering, and transformation
- **Key operations:** `read_csv()`, `dropna()`, `value_counts()`, `groupby()`

```python
import numpy as np
```
- **Purpose:** Numerical computing library
- **Used for:** Array operations, mathematical functions, sorting indices
- **Key operations:** `np.arange()`, `np.argsort()`, `np.linspace()`

### Visualization

```python
import matplotlib.pyplot as plt
```
- **Purpose:** Primary plotting library
- **Used for:** Creating all charts and visualizations
- **Key operations:** `subplots()`, `bar()`, `barh()`, `scatter()`, `savefig()`

```python
import seaborn as sns
```
- **Purpose:** Statistical data visualization (built on matplotlib)
- **Used for:** Creating heatmaps for confusion matrices
- **Key operations:** `heatmap()` with annotations

### Machine Learning - Model Selection & Evaluation

```python
from sklearn.model_selection import train_test_split
```
- **Purpose:** Split data into training and testing sets
- **Used for:** Creating 80/20 train-test split with stratification
- **Parameters:** `test_size=0.2`, `random_state=42`, `stratify=y`

```python
from sklearn.feature_extraction.text import TfidfVectorizer
```
- **Purpose:** Convert text to numerical features using TF-IDF
- **How it works:**
  - **TF (Term Frequency):** How often a word appears in a document
  - **IDF (Inverse Document Frequency):** How rare/important a word is across all documents
  - **TF-IDF = TF × IDF:** Balances frequency with importance
- **Parameters used:**
  - `max_features=10000`: Keep top 10,000 most important words
  - `ngram_range=(1, 2)`: Use single words (unigrams) and word pairs (bigrams)
  - `min_df=5`: Ignore words appearing in fewer than 5 documents
  - `max_df=0.95`: Ignore words appearing in more than 95% of documents
  - `stop_words='english'`: Remove common English words (the, is, at, etc.)

### Machine Learning - Classifiers

```python
from sklearn.linear_model import LogisticRegression
```
- **Purpose:** Linear classifier for binary/multiclass classification
- **How it works:** Finds a linear decision boundary using logistic function
- **Strengths:** Fast, interpretable, works well with high-dimensional sparse data (like text)
- **Parameters:** `max_iter=1000`, `random_state=42`, `n_jobs=-1`

```python
from sklearn.naive_bayes import MultinomialNB
```
- **Purpose:** Probabilistic classifier based on Bayes' theorem
- **How it works:** Assumes features are conditionally independent given the class
- **Strengths:** Very fast, works well with text data, good baseline
- **Parameters:** `alpha=0.1` (Laplace smoothing)

```python
from sklearn.svm import LinearSVC
```
- **Purpose:** Support Vector Machine with linear kernel
- **How it works:** Finds the hyperplane that maximizes margin between classes
- **Strengths:** Effective in high-dimensional spaces, memory efficient
- **Parameters:** `random_state=42`, `max_iter=2000`

```python
from sklearn.ensemble import RandomForestClassifier
```
- **Purpose:** Ensemble of decision trees
- **How it works:** Trains multiple decision trees and aggregates their predictions
- **Strengths:** Handles non-linear relationships, resistant to overfitting
- **Parameters:** `n_estimators=100`, `random_state=42`, `n_jobs=-1`

### Machine Learning - Metrics

```python
from sklearn.metrics import (
    accuracy_score,      # Overall correctness: (TP + TN) / Total
    precision_score,     # Of predicted positives, how many are correct: TP / (TP + FP)
    recall_score,        # Of actual positives, how many were found: TP / (TP + FN)
    f1_score,           # Harmonic mean of precision and recall
    classification_report,  # Detailed report with all metrics per class
    confusion_matrix,    # Matrix showing TP, TN, FP, FN
    roc_curve,          # Receiver Operating Characteristic curve data
    auc                 # Area Under the ROC Curve
)
```

### Machine Learning - Pipeline

```python
from sklearn.pipeline import Pipeline
```
- **Purpose:** Chain multiple processing steps together
- **Used for:** Combining TF-IDF vectorization with classifiers
- **Benefit:** Ensures consistent preprocessing during training and prediction

### Utility Libraries

```python
import warnings
warnings.filterwarnings('ignore')
```
- **Purpose:** Suppress warning messages for cleaner output

```python
import re
```
- **Purpose:** Regular expressions for text pattern matching
- **Used for:** Removing URLs, HTML tags, punctuation from text

```python
import string
```
- **Purpose:** String constants and utilities
- **Used for:** Access to punctuation characters (imported but not directly used)

```python
import joblib
```
- **Purpose:** Efficient serialization of Python objects
- **Used for:** Saving and loading trained models to/from disk

```python
import argparse
```
- **Purpose:** Command-line argument parsing
- **Used for:** Implementing different execution modes (train, ui, demo)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SENTIMENT ANALYSIS PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: DATA LOADING                                                         │
│ ─────────────────────                                                        │
│ • Load CSV dataset (1.44M reviews)                                          │
│ • Optional: Sample 100,000 reviews for faster processing                    │
│ • Output: Raw DataFrame                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: EXPLORATORY DATA ANALYSIS                                           │
│ ─────────────────────────────────────                                       │
│ • Display dataset shape and columns                                         │
│ • Check for missing values                                                  │
│ • Show sentiment distribution (POSITIVE vs NEGATIVE)                        │
│ • Calculate text statistics (length, word count)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: DATA VISUALIZATION                                                   │
│ ──────────────────────────                                                  │
│ • Sentiment Distribution Bar Chart                                          │
│ • Review Length Box Plot by Sentiment                                       │
│ • Word Count Histogram by Sentiment                                         │
│ • Sentiment by Critic Type (Top Critic vs Regular)                          │
│ • Output: data/sentiment_analysis_eda.png                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: TEXT PREPROCESSING                                                   │
│ ──────────────────────────                                                  │
│ • Remove rows with missing reviewText or scoreSentiment                     │
│ • Keep only POSITIVE and NEGATIVE sentiments                                │
│ • Clean text:                                                               │
│   - Convert to lowercase                                                    │
│   - Remove URLs (http://, https://, www.)                                   │
│   - Remove HTML tags (<...>)                                                │
│   - Remove punctuation (except ! and ?)                                     │
│   - Remove extra whitespace                                                 │
│ • Create binary labels (1 = POSITIVE, 0 = NEGATIVE)                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: TRAIN-TEST SPLIT                                                     │
│ ─────────────────────────                                                   │
│ • Split data: 80% training, 20% testing                                     │
│ • Stratified split (maintains class proportions)                            │
│ • Random state: 42 (for reproducibility)                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: MODEL TRAINING                                                       │
│ ──────────────────────                                                      │
│ For each model (Logistic Regression, Naive Bayes, Linear SVM, Random Forest):│
│ • Create Pipeline: TfidfVectorizer → Classifier                             │
│ • Fit on training data                                                      │
│ • Predict on test data                                                      │
│ • Calculate metrics (Accuracy, Precision, Recall, F1)                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 7: RESULTS VISUALIZATION                                                │
│ ─────────────────────────────                                               │
│ • Model Performance Comparison (grouped bar chart)                          │
│ • Confusion Matrix for best model (heatmap)                                 │
│ • F1 Score Comparison (horizontal bar chart)                                │
│ • Model Accuracy Overview (scatter plot)                                    │
│ • Output: data/model_comparison.png                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 8: FEATURE IMPORTANCE                                                   │
│ ──────────────────────────                                                  │
│ • Extract TF-IDF feature names and Logistic Regression coefficients        │
│ • Identify top 20 words for POSITIVE sentiment (highest coefficients)       │
│ • Identify top 20 words for NEGATIVE sentiment (lowest coefficients)        │
│ • Create visualization                                                      │
│ • Output: data/feature_importance.png                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 9: SAVE BEST MODEL                                                      │
│ ───────────────────────                                                     │
│ • Identify best model by F1 score                                           │
│ • Serialize and save to data/best_model.joblib                              │
│ • Model can be loaded later for predictions                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Detailed Function Breakdown

### 1. Data Loading Functions

#### `load_data(filepath: str, sample_size: int = None) -> pd.DataFrame`

**Purpose:** Load the movie reviews dataset from a CSV file.

**Parameters:**
- `filepath`: Path to the CSV file
- `sample_size`: Number of reviews to sample (None for full dataset)

**Process:**
1. Read CSV file using pandas
2. If sample_size is specified, randomly sample that many rows
3. Print loading status

**Returns:** DataFrame with movie reviews

---

### 2. Exploratory Data Analysis Functions

#### `explore_data(df: pd.DataFrame) -> None`

**Purpose:** Perform and display exploratory data analysis.

**Outputs to console:**
- Dataset shape (rows × columns)
- Column names
- Missing values count for reviewText and scoreSentiment
- Sentiment distribution (counts and percentages)
- Text statistics:
  - Average character length
  - Average word count
  - Min/Max word counts

#### `visualize_data(df: pd.DataFrame) -> None`

**Purpose:** Create a 2×2 grid of visualizations.

**Visualizations created:**

| Position | Chart Type | Description |
|----------|-----------|-------------|
| Top-Left | Bar Chart | Sentiment distribution with percentages |
| Top-Right | Box Plot | Review character length by sentiment |
| Bottom-Left | Histogram | Word count distribution by sentiment |
| Bottom-Right | Grouped Bar | Sentiment by critic type (Top vs Regular) |

**Output file:** `data/sentiment_analysis_eda.png`

**Color scheme:**
- 🟢 Green (`#2ecc71`): POSITIVE sentiment
- 🔴 Red (`#e74c3c`): NEGATIVE sentiment

---

### 3. Text Preprocessing Functions

#### `clean_text(text: str) -> str`

**Purpose:** Clean and normalize review text for machine learning.

**Cleaning steps:**
1. Handle NaN values → return empty string
2. Convert to string type
3. Convert to lowercase
4. Remove URLs (http, https, www patterns)
5. Remove HTML tags
6. Remove punctuation (except ! and ? which carry sentiment)
7. Remove extra whitespace

**Example:**
```
Input:  "This movie was AMAZING!!! Check it out at http://example.com <br>"
Output: "this movie was amazing!"
```

#### `preprocess_data(df: pd.DataFrame) -> pd.DataFrame`

**Purpose:** Prepare the entire dataset for model training.

**Steps:**
1. Remove rows with missing reviewText or scoreSentiment
2. Filter to only POSITIVE and NEGATIVE sentiments
3. Apply `clean_text()` to all reviews
4. Remove reviews that become empty after cleaning
5. Create binary labels: POSITIVE=1, NEGATIVE=0

**Returns:** Cleaned DataFrame with new columns:
- `cleaned_text`: Preprocessed review text
- `sentiment_label`: Binary label (0 or 1)

---

### 4. Model Training Functions

#### `create_models() -> dict`

**Purpose:** Create dictionary of model pipelines to train.

**Models created:**

| Model | TF-IDF Features | Classifier Parameters |
|-------|-----------------|----------------------|
| Logistic Regression | 10,000 | max_iter=1000, n_jobs=-1 |
| Naive Bayes | 10,000 | alpha=0.1 |
| Linear SVM | 10,000 | max_iter=2000 |
| Random Forest | 5,000 | n_estimators=100, n_jobs=-1 |

**TF-IDF Configuration (shared):**
- `ngram_range=(1, 2)`: Unigrams and bigrams
- `min_df=5`: Minimum document frequency
- `max_df=0.95`: Maximum document frequency
- `stop_words='english'`: Remove English stop words

#### `train_and_evaluate(...) -> dict`

**Purpose:** Train all models and evaluate their performance.

**Process for each model:**
1. Fit pipeline on training data (X_train, y_train)
2. Predict on test data (X_test)
3. Calculate metrics:
   - **Accuracy:** Overall correctness
   - **Precision:** Positive predictive value
   - **Recall:** True positive rate (sensitivity)
   - **F1-Score:** Harmonic mean of precision and recall

**Returns:** Dictionary containing:
```python
{
    'Model Name': {
        'model': trained_pipeline,
        'accuracy': float,
        'precision': float,
        'recall': float,
        'f1': float,
        'y_pred': array
    }
}
```

---

### 5. Visualization Functions

#### `plot_results(results: dict, y_test: pd.Series) -> None`

**Purpose:** Create model comparison visualizations.

**Visualizations:**

| Position | Chart | Description |
|----------|-------|-------------|
| Top-Left | Grouped Bar | All metrics for all models |
| Top-Right | Heatmap | Confusion matrix for best model |
| Bottom-Left | Horizontal Bar | F1 scores ranked |
| Bottom-Right | Scatter | Accuracy overview with color gradient |

**Output file:** `data/model_comparison.png`

#### `print_classification_reports(results: dict, y_test: pd.Series) -> None`

**Purpose:** Print detailed sklearn classification reports.

**Output per model:**
```
              precision    recall  f1-score   support

    NEGATIVE       0.75      0.55      0.63      6297
    POSITIVE       0.80      0.91      0.85     12742

    accuracy                           0.79     19039
   macro avg       0.78      0.73      0.74     19039
weighted avg       0.79      0.79      0.78     19039
```

---

### 6. Feature Importance Functions

#### `show_feature_importance(results: dict, top_n: int = 20) -> None`

**Purpose:** Identify and visualize most important words for sentiment.

**How it works:**
1. Extract TF-IDF vocabulary (feature names)
2. Extract Logistic Regression coefficients
3. Sort coefficients to find:
   - **Highest coefficients** → POSITIVE indicator words
   - **Lowest coefficients** → NEGATIVE indicator words

**Console output:**
```
Top 20 words indicating POSITIVE sentiment:
----------------------------------------
  entertaining         (coef: 4.2430)
  enjoyable            (coef: 4.0315)
  ...

Top 20 words indicating NEGATIVE sentiment:
----------------------------------------
  fails                (coef: -5.9928)
  unfortunately        (coef: -5.4558)
  ...
```

**Output file:** `data/feature_importance.png`

---

### 7. Prediction Functions

#### `predict_sentiment(model, text: str) -> dict`

**Purpose:** Predict sentiment for a single review.

**Process:**
1. Clean the input text
2. Use model to predict class
3. Get probability scores (if available)
4. Format result

**Returns:**
```python
{
    'text': 'truncated review text...',
    'sentiment': 'POSITIVE' or 'NEGATIVE',
    'confidence': 0.95  # or None for SVM
}
```

#### `interactive_demo(model) -> None`

**Purpose:** Run command-line demo with sample predictions.

**Sample reviews tested:**
1. "This movie was absolutely fantastic!..." → Expected: POSITIVE
2. "Terrible film. Waste of time..." → Expected: NEGATIVE
3. "A decent movie with some good moments..." → Could be either
4. "One of the best films I've ever seen..." → Expected: POSITIVE
5. "I couldn't even finish watching this garbage..." → Expected: NEGATIVE

---

### 8. Model Persistence Functions

#### `save_best_model(results: dict, filepath: str) -> None`

**Purpose:** Save the best performing model to disk.

**Selection criteria:** Highest F1 score

**Output file:** `data/best_model.joblib`

**Usage for loading:**
```python
model = joblib.load('data/best_model.joblib')
prediction = model.predict(["Your review text here"])
```

---

### 9. Interactive UI Function

#### `run_interactive_ui() -> None`

**Purpose:** Launch a web-based interface for sentiment prediction.

**Technology:** Gradio (Python web UI library)

**Features:**
- Text input area for entering reviews
- "Analyze Sentiment" button
- Results display with:
  - Prediction (POSITIVE ✅ or NEGATIVE ❌)
  - Confidence percentage
  - Probability breakdown bar
- Example reviews to try
- Model information panel

**URL:** `http://127.0.0.1:7860`

---

## Execution Modes

The application supports 3 execution modes via command-line arguments:

### Mode 1: Train (`--mode train`)

```bash
python app.py --mode train
```

**What it does:**
1. Loads and explores the dataset
2. Creates visualizations (EDA)
3. Preprocesses text data
4. Trains 4 different models
5. Evaluates and compares models
6. Shows feature importance
7. Runs demo predictions
8. Saves the best model

**Output files generated:**
- `data/sentiment_analysis_eda.png`
- `data/model_comparison.png`
- `data/feature_importance.png`
- `data/best_model.joblib`

**Default mode** (runs if no --mode specified)

---

### Mode 2: UI (`--mode ui`)

```bash
python app.py --mode ui
```

**What it does:**
1. Loads the saved best model
2. Launches Gradio web interface
3. Opens browser automatically

**Requirements:**
- Must run `--mode train` first to create the model
- Gradio package must be installed

**Access:** Open `http://127.0.0.1:7860` in browser

---

### Mode 3: Demo (`--mode demo`)

```bash
python app.py --mode demo
```

**What it does:**
1. Loads the saved best model
2. Runs command-line interactive demo
3. Shows predictions for sample reviews

**Requirements:**
- Must run `--mode train` first to create the model

---

## Output Files

### 1. `data/sentiment_analysis_eda.png`

**Type:** Exploratory Data Analysis visualization (2×2 grid)

**Contents:**
- Sentiment distribution with percentages
- Review length distribution by sentiment
- Word count distribution by sentiment
- Sentiment breakdown by critic type

**Size:** ~300 DPI, high quality for reports

---

### 2. `data/model_comparison.png`

**Type:** Model evaluation visualization (2×2 grid)

**Contents:**
- All metrics comparison (accuracy, precision, recall, F1)
- Confusion matrix for best model
- F1 score ranking
- Accuracy scatter overview

---

### 3. `data/feature_importance.png`

**Type:** Feature importance visualization (1×2 layout)

**Contents:**
- Top 10 words indicating POSITIVE sentiment
- Top 10 words indicating NEGATIVE sentiment

---

### 4. `data/best_model.joblib`

**Type:** Serialized scikit-learn Pipeline

**Contents:**
- Complete trained pipeline including:
  - TfidfVectorizer (with fitted vocabulary)
  - Classifier (with trained weights)

**Size:** Varies (~10-50 MB depending on vocabulary size)

**Loading:**
```python
import joblib
model = joblib.load('data/best_model.joblib')
result = model.predict(["Great movie!"])  # Returns [1] for POSITIVE
```

---

## Model Performance

### Typical Results (on 100,000 sample):

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Logistic Regression** | 79.06% | 80.27% | 91.11% | **85.35%** |
| Naive Bayes | 77.19% | 77.17% | 93.63% | 84.60% |
| Linear SVM | 78.30% | 81.42% | 87.55% | 84.38% |
| Random Forest | 75.36% | 77.71% | 88.60% | 82.80% |

### Best Model: Logistic Regression

**Why Logistic Regression performs best:**
1. Works well with high-dimensional sparse data (TF-IDF vectors)
2. Linear decision boundary is appropriate for text classification
3. Fast training and prediction
4. Highly interpretable (coefficients show feature importance)

### Key Observations:
- All models favor POSITIVE predictions (higher recall for positive)
- This is due to class imbalance (~67% positive, ~33% negative)
- Logistic Regression has the best balance of precision and recall

---

## How to Use

### Installation

```bash
# Install required packages
pip install pandas numpy scikit-learn matplotlib seaborn joblib gradio
```

### Training the Model

```bash
# Navigate to project directory
cd movie-recomendation-system-ml

# Run training (default mode)
python app.py
# or explicitly
python app.py --mode train
```

### Using the Web Interface

```bash
# First, ensure model is trained
python app.py --mode train

# Then launch UI
python app.py --mode ui
```

### Making Predictions in Code

```python
import joblib
from app import clean_text

# Load model
model = joblib.load('data/best_model.joblib')

# Predict
review = "This movie was absolutely wonderful!"
cleaned = clean_text(review)
prediction = model.predict([cleaned])[0]
probability = model.predict_proba([cleaned])[0]

print(f"Sentiment: {'POSITIVE' if prediction == 1 else 'NEGATIVE'}")
print(f"Confidence: {max(probability):.1%}")
```

### Configuration Options

In the `main()` function, you can modify:

```python
DATA_PATH = 'data/dataset.csv'  # Path to dataset
SAMPLE_SIZE = 100000            # Set to None for full dataset
TEST_SIZE = 0.2                 # Train/test split ratio
RANDOM_STATE = 42               # For reproducibility
```

---

## Summary

This project demonstrates a complete machine learning pipeline for sentiment analysis:

1. **Data Engineering:** Loading, cleaning, and preprocessing text data
2. **Feature Engineering:** TF-IDF vectorization with unigrams and bigrams
3. **Model Training:** Comparison of 4 different algorithms
4. **Evaluation:** Comprehensive metrics and visualizations
5. **Interpretation:** Feature importance analysis
6. **Deployment:** Web UI for real-time predictions
7. **Persistence:** Model serialization for production use

The best performing model (Logistic Regression) achieves **85% F1-Score**, making it suitable for practical sentiment classification tasks.
