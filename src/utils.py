import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get classification metrics for the test data
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted')
            recall = recall_score(y_test, y_test_pred, average='weighted')
            f1 = f1_score(y_test, y_test_pred, average='weighted')
            try:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            except:
                roc_auc = np.nan  # Handle cases where ROC-AUC cannot be computed

            # Store the metrics in the report dictionary
            report[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'ROC-AUC Score': roc_auc
            }
        return report
    
    except Exception as e:
        logging.error(f'Exception occurred during model evaluation: {e}')
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            obj = pickle.load(file_obj)
        logging.info(f'Successfully loaded object from {file_path}')
        return obj
    except Exception as e:
        logging.error(f'Exception occurred while loading object from {file_path}: {e}')
        raise CustomException(e, sys)