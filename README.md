# Leveraging Machine Learning for Automated Consumer Complaint Classification

## Table of Contents

- [Overview](#overview)
- [Workflow](#workflow)
- [Installation](#installation)


## Overview
This project aims to automate the classification of consumer complaints into predefined categories using various machine learning algorithms such as RandomForest, XGBoost, Bernoulli Naive Bayes, DecisionTree, CatBoost, RandomForestClassifier, XGBClassifier, and LightGBM. 

## Workflow
1. **Data Preparation**: Load the data, handle missing values, and carry out preliminary data analysis.
2. **Text Preprocessing**: Convert the complaint text into a matrix of TF-IDF features using TF-IDF Vectorizer.
3. **Training Models**: Train a variety of machine learning models for the classification task.
4. **Model Evaluation**: Evaluate each model based on its accuracy using precision, recall, and F1-score.
5. **Hyperparameter Tuning**: Perform hyperparameter tuning on the RandomForest and XGBClassifier models using RandomizedSearchCV.
6. **Stacking Classifier**: Build a stacking classifier that includes RandomForest, Bernoulli Naive Bayes, and XGBClassifier.

## Installation
Clone the repository and install the dependencies:

pip install -r requirements.txt
