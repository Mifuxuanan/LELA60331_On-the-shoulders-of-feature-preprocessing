# LELA60331_On-the-shoulders-of-feature-preprocessing
A coursework code related to an NLP system for the classification of product reviews
# Multi-Task Text Classification with Logistic Regression

## Overview
This project implements a logistic regression-based multi-task classifier on the following three datasets of Amazon Reviews:
- **Sentiment Analysis**: Classify positive or negative sentiment labels.
- **Product Types Classification**: Identify 24 product types based on the reviews.
- **Helpfulness Analysis**: Recognize reviews as helpful, unhelpful, or neutral.

The classifier uses a modular design and supports parameter-based task switching. It evaluates the model's performance with various preprocessing methods (e.g., unigram/bigram tokenization, One-hot/TF-IDF encodings, and stopwords filtering).

---
## Keypoint
- **Multi-task Support**: One model for three classification tasks:
  - Sentiment analysis with the dataset of sentiment_ratings
  - Product types classification with the dataset of product_types
  - Helpfulness analysis with the dataset of helpfulness_ratings 
- **Diverse Feature Preprocessing Methods**: Supports individual and combination applications of the following methods:
  - Unigram or Bigram tokenization.
  - Use of TF-IDF or binary feature encoding.
  - Stopwords filtering.
- **Model evaluation**: Evaluate preprocessing combinations' impact on precision, recall, and other metrics.
---
## File Structure
- Compiled_Reviews.txt # Input data 
- main() # Main function for task execution 
- modules/ # Encapsulated preprocessing and model training functions
- results/ # Outputs (the first five dimensions of the weight matrix, metrics, plots, etc.)
---
## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- math
- re

## Usage
## Run the Classifier
The main function supports parameter-based control. Example:

```python
from main import main

precision, recall, p_macro, r_macro = main(
    label_data=sentiment_ratings,
    apply_stopwords_list=True,
    n=2,
    TF_IDF=True,
    k=32,
    lr=0.001,
    n_iters=200,
    eval_set="test"
)
```

---

## Results
Evaluation Metrics: Precision, Recall, Macro-average Precision/Recall.
Weight Interpretation: Top 5 weights of the logistic regression model for each task.
Evaluation of performance: Effectiveness of preprocessing combinations visualized via plots.

Developed by [Mei Jia].
