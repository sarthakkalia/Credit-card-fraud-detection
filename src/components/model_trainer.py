# Basic Import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'Logistic Regression':LogisticRegression(),
            'Naive Bayes':GaussianNB(),
            'Decision Tree':DecisionTreeClassifier(),
            'Random Forest':RandomForestClassifier(),
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # # Extract the best model based on the ROC-AUC Score
            # best_model_score = max(model['ROC-AUC Score'] for model in model_report.values())
            # best_model_name = [name for name, metrics in model_report.items() if metrics['ROC-AUC Score'] == best_model_score][0]

            # Extract the best model based on Accuracy
            best_model_name = max(model_report, key=lambda x: model_report[x]['Accuracy'])
            best_model_score = model_report[best_model_name]['Accuracy']

            best_model = models[best_model_name]

            print(f'Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found, Model Name: {best_model_name}, Accuracy: {best_model_score}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

          
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)