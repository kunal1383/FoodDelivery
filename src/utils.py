import os
import sys
import pickle
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from src.exception import CustomException
from src.logger import logging

import datetime


MONGO_USERNAME=''
MONGO_PASSWORD=''
MONGO_DB='FoodDelivery'
COLLECTION_NAME='ProcessedData'


def frac_to_time(df, column_name: str):
    for i, value in enumerate(df[column_name]):
        # checking if we are dealing with a string and is a fraction time
        if isinstance(value, str) and '.' in value:
            fractional_hours = float(value)
            # Convert fractional hours to seconds
            total_seconds = fractional_hours * 24 * 60 * 60  
            # we are getting extra decimal values so dealing with them
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            time_value = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            # Replace the fractional value with the valid time value
            df.at[i, column_name] = time_value
    return df        


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:

        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        # # Hyperparameters for DecisionTree
        # param_dist_decision_tree = {
        #     'criterion': ['squared_error'], 
        #     'splitter': ['best'],  
        #     'max_depth': [None, 10, 20],  
        #     'min_samples_split': [2, 5],  
        #     'min_samples_leaf': [1, 2] 
        # }

        # # Hyperparameters for RandomForest
        # param_dist_random_forest = {
        #     'n_estimators': [100, 200],  
        #     'criterion': ['squared_error'],  
        #     'max_depth': [None, 10],  
        #     'min_samples_split': [2, 5],  
        #     'min_samples_leaf': [1, 2] 
        # }

        for model_name, model in models.items():
            # if model_name == 'DecisionTree':
            #     #RandomizedSearchCV instance for DecisionTree
            #     random_search = RandomizedSearchCV(model, param_distributions=param_dist_decision_tree, n_iter=10, cv=5)
            #     # Fitting RandomizedSearchCV on training data
            #     random_search.fit(X_train, y_train)
                
            #     best_model = random_search.best_estimator_
                
            #     y_test_pred = best_model.predict(X_test)
            # elif model_name == 'RandomForest':
            #     #RandomizedSearchCV instance for RandomForest
            #     random_search = RandomizedSearchCV(model, param_distributions=param_dist_random_forest, n_iter=10, cv=5)
            #     # Fitting RandomizedSearchCV on training data
            #     random_search.fit(X_train, y_train)
                
            #     best_model = random_search.best_estimator_
                
            #     y_test_pred = best_model.predict(X_test)
            # else:
            #     # For other models, proceeding with regular fitting and prediction
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 score for the model
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report

    except Exception as e:
        logging.info('Exception occurred during model training')
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    
    

def parse_date(date_str):
    # Convert the date string to datetime format
    date_obj = pd.to_datetime(date_str, format='%d-%m-%Y')
    # Extract year, month, and day from the datetime object
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day
    return year, month, day

def parse_time(time_str):
    # Convert the time string to datetime format
    time_obj = pd.to_datetime(time_str, format='%H:%M')
    # Extract hour and minute from the datetime object
    hour = time_obj.hour
    minute = time_obj.minute
    return hour, minute    
