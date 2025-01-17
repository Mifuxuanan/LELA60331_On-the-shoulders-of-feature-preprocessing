# LELA60331_On-the-shoulders-of-feature-preprocessing
A coursework code related to an NLP system for the classification of product reviews
# Multi-Task Text Classification with Logistic Regression

## Overview
This project implements a logistic regression-based multi-task classifier designed to handle:
- **Sentiment Analysis**: Classify positive, negative, or neutral product reviews.
- **Product Category Classification**: Identify product categories based on textual reviews.
- **Helpfulness Analysis**: Classify reviews as helpful, unhelpful, or neutral.

The classifier leverages a modular design and supports parameter-based task switching. It evaluates the performance impact of various preprocessing methods (e.g., unigram/bigram, TF-IDF, stopwords filtering).

---
## Features
- **Multi-task Support**: One model for three classification tasks.
- **Preprocessing Flexibility**: Supports combinations of:
  - Unigram or Bigram tokenization.
  - Use of TF-IDF or binary feature encoding.
  - Stopwords filtering.
- **Comparative Analysis**: Evaluate preprocessing combinations' impact on precision, recall, and other metrics.
---
## File Structure
├── Compiled_Reviews.txt # Input data ├── main.py # Main script for task execution ├── modules/ # Encapsulated preprocessing and model training functions └── results/ # Output files (metrics, plots, etc.)
---
## Requirements
- Python 3.8+
- NumPy
- Matplotlib
- re

Install dependencies using:
```bash
pip install -r requirements.txt
---

## Usage
## Run the Classifier
The main function supports parameter-based control. Example:

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

---

## Results
Evaluation Metrics: Precision, Recall, Macro-average Precision/Recall.
Weight Interpretation: Top 5 weights of the logistic regression model for each task.
Comparative Performance: Effectiveness of preprocessing combinations visualized via plots.

Developed by [Mei Jia].
