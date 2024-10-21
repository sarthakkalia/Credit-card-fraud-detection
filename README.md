# Credit Card Fraud Detection - Machine Learning Pipeline

This project focuses on detecting fraudulent credit card transactions using various machine learning models. The aim is to build a robust pipeline to handle data preprocessing, training, and evaluation of models to select the best-performing one for fraud detection.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling Approach](#modeling-approach)
- [Results](#results)
- [References](#references)
- [Contributors](#contributors)

## Project Overview
Credit card fraud is a major concern, and machine learning models can help detect such activities by learning patterns from historical data. This project implements a machine learning pipeline for fraud detection, evaluating various classifiers to identify the best-performing model based on accuracy and ROC-AUC score.

## Dataset Information
- **Source**: Credit Card Fraud Detection dataset.
- The dataset contains several features such as demographic details, transaction history, and repayment amounts to predict whether a customer will default on the payment for the next month.

### Features:
- `LIMIT_BAL`: Amount of the given credit (in NT dollars).
- `SEX`: Gender (1 = male, 2 = female).
- `EDUCATION`: Education level (1 = graduate school, 2 = university, 3 = high school, 4 = others).
- `MARRIAGE`: Marital status (1 = married, 2 = single).
- `AGE`: Age of the client.
- `PAY_0` to `PAY_6`: History of past payment records.
- `BILL_AMT1` to `BILL_AMT6`: Amount of bill statement (in NT dollars).
- `PAY_AMT1` to `PAY_AMT6`: Amount of previous payments (in NT dollars).
- **Target Variable**: `default payment next month` (1 = default, 0 = not default).

## Project Structure

```
.
├── artifacts
│   ├── model.pkl                 # Trained model
│   ├── preprocessor.pkl           # Preprocessing object
│   ├── raw.csv                    # Raw data
│   ├── train.csv                  # Training data
│   └── test.csv                   # Test data
├── notebooks
│   ├── data
│   │   └── creditCardFraud_28011964_120214.csv   # Original dataset
│   ├── credit-card-fraud-detection_EDA.ipynb     # Exploratory Data Analysis
├── src
│   ├── components
│   │   ├── data_ingestion.py      # Data ingestion component
│   │   ├── data_transformation.py # Data transformation component
│   │   └── model_trainer.py       # Model trainer component
│   ├── pipelines
│   │   ├── prediction_pipeline.py # Pipeline for prediction
│   │   └── training_pipeline.py   # Pipeline for training
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging configuration
│   └── utils.py                   # Utility functions
├── templates
│   ├── form.html                  # HTML form for user input
│   └── index.html                 # Home page HTML
├── application.py                 # Flask application entry point
├── requirements.txt               # Required Python packages
├── setup.py                       # Package setup
└── .gitignore                     # Git ignore file
```

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/sarthakkalia/Credit-card-fraud-detection.git
```

### 2. Navigate to the project directory
```bash
cd Credit-card-fraud-detection
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the application
```bash
python application.py
```

## Usage
The Flask web application serves predictions. Upload customer data through the form to get predictions on whether the customer is likely to default on their payment.

## Modeling Approach
### Models used:
- `LogisticRegression`
- `GaussianNB`
- `DecisionTreeClassifier`
- `RandomForestClassifier`

Each model is evaluated based on Accuracy and ROC-AUC score. We implemented a machine learning pipeline to automate the process of data ingestion, transformation, model training, and evaluation.

### Model Selection:
After training multiple models, the best one is selected based on the Accuracy score, which is then used for fraud prediction.

#### Code for selecting the best model:
```python
best_model_name = max(model_report, key=lambda x: model_report[x]['Accuracy'])
best_model_score = model_report[best_model_name]['Accuracy']

best_model = models[best_model_name]
print(f'Best Model: {best_model_name}, Accuracy: {best_model_score}')
```

For detailed information on the pipeline, check out the [GitHub repository](https://github.com/sarthakkalia/Machine-Learning-Pipeline).

## Results
- **Best Performing Model**: `RandomForestClassifier`
- The best model achieved high accuracy and ROC-AUC score, making it reliable for detecting fraudulent transactions.

## References
- Dataset: Credit Card Fraud Detection dataset.

## Contributors
- **Sarthak Kumar Kalia** - [LinkedIn](https://www.linkedin.com/in/sarthak-kalia/)
