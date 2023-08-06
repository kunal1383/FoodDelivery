import os
import sys
import pandas as pd 
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import frac_to_time ,save_object
from src.mongodb import MongoDBHandler

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import OrdinalEncoder ,OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


from dataclasses import dataclass


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformation_object(self):
        try:
            
            logging.info('Data Transformation initiated')
            #Separating the numerical ,categorical and ordinal
            categorical_cols = ['Weather_conditions', 'Type_of_vehicle', 'Festival', 'City']
            numerical_cols = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
            'Restaurant_longitude', 'Delivery_location_latitude',
            'Delivery_location_longitude', 'Vehicle_condition',
            'multiple_deliveries', 'year', 'month', 'day', 'Time_Orderd_hour',
            'Time_Orderd_minute', 'Time_Order_picked_hour',
            'Time_Order_picked_minute']
            ordinal_col = 'Road_traffic_density'
            ordinal_cols = [ordinal_col] 
            
            # Custom ranking for Road_traffic_density column
            traffic_density = ['Low', 'Medium', 'High', 'Jam']

            logging.info('Pipeline Initiated')
            
            # Creating the transformers for numerical, ordinal, and nominal data
            num_transformer = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler', StandardScaler(with_mean=False))

                ]
            )

            ordinal_transformer = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories='auto')),
                ('scaler',StandardScaler())
                ]
            )

            nominal_transformer = Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
                ]
            
            )

            # Creating the column transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_transformer', num_transformer, numerical_cols),
                    ('ordinal_transformer', ordinal_transformer, ordinal_cols),
                    ('nominal_transformer', nominal_transformer, categorical_cols)
                ]
            )       
            logging.info('Pipeline Completed')
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        
            logging.info('Read train and test data completed')
            
            # Converting the string date to datetime in train_df
            logging.info('Converting the date column')
            
            train_df['Order_Date'] = pd.to_datetime(train_df['Order_Date'], format='%d-%m-%Y')
            train_df['year'] = train_df['Order_Date'].dt.year
            train_df['month'] = train_df['Order_Date'].dt.month
            train_df["day"] = train_df["Order_Date"].dt.day
            
            # Converting the string date to datetime in test_df
            test_df['Order_Date'] = pd.to_datetime(test_df['Order_Date'], format='%d-%m-%Y')

            test_df['year'] = test_df['Order_Date'].dt.year
            test_df['month'] = test_df['Order_Date'].dt.month
            test_df["day"] = test_df["Order_Date"].dt.day
            logging.info('Completed  the date column transformation.')
            
            logging.info("Converting the fraction values in Time columns into datetime format")
            # converting the Time_order and Time_ordered_picked to datetime format
            train_df = frac_to_time(train_df, 'Time_Orderd')
            train_df = frac_to_time(train_df, 'Time_Order_picked')

            test_df = frac_to_time(test_df, 'Time_Orderd')
            test_df = frac_to_time(test_df, 'Time_Order_picked')
            logging.info('Completed  the fraction to Time columns transformation.')
            
            logging.info("Converting the  Time columns into datetime format")
            # Convert 'Time_Orderd' column to datetime in train_df
            train_df['Time_Orderd'] = pd.to_datetime(train_df['Time_Orderd'], errors='coerce')
            train_df['Time_Order_picked'] = pd.to_datetime(train_df['Time_Order_picked'], errors='coerce')
            

            # Convert 'Time_Orderd' column to datetime in test_df
            test_df['Time_Orderd'] = pd.to_datetime(test_df['Time_Orderd'], errors='coerce')
            test_df['Time_Order_picked'] = pd.to_datetime(test_df['Time_Order_picked'], errors='coerce')
            
            logging.info('Completed  the Time columns transformation.')
            
            
            logging.info('Filling missing values in time columns')
            # Filling the null values in 'Time_Order_picked' and 'Time_Orderd' columns with most frequent values in train_df
            most_frequent_picked_time_train = train_df['Time_Order_picked'].mode()[0]
            most_frequent_orderd_time_train = train_df['Time_Orderd'].mode()[0]

            train_df['Time_Order_picked'] = train_df['Time_Order_picked'].fillna(most_frequent_picked_time_train)
            train_df['Time_Orderd'] = train_df['Time_Orderd'].fillna(most_frequent_orderd_time_train)

            # Filling the null values in 'Time_Order_picked' and 'Time_Orderd' columns with most frequent values in test_df
            most_frequent_picked_time_test = test_df['Time_Order_picked'].mode()[0]
            most_frequent_orderd_time_test = test_df['Time_Orderd'].mode()[0]

            test_df['Time_Order_picked'] = test_df['Time_Order_picked'].fillna(most_frequent_picked_time_test)
            test_df['Time_Orderd'] = test_df['Time_Orderd'].fillna(most_frequent_orderd_time_test)
            
            logging.info('Completed filling missing values in time columns')

            logging.info("Separating the hours and minutes into new columns")
            
            train_df['Time_Orderd_hour'] = train_df['Time_Orderd'].dt.hour.astype(int)
            train_df['Time_Orderd_minute'] = train_df['Time_Orderd'].dt.minute.astype(int)

            train_df['Time_Order_picked_hour'] = train_df['Time_Order_picked'].dt.hour.astype(int)
            train_df['Time_Order_picked_minute'] = train_df['Time_Order_picked'].dt.minute.astype(int)

            # For test_df
            test_df['Time_Orderd_hour'] = test_df['Time_Orderd'].dt.hour.astype(int)
            test_df['Time_Orderd_minute'] = test_df['Time_Orderd'].dt.minute.astype(int)

            test_df['Time_Order_picked_hour'] = test_df['Time_Order_picked'].dt.hour.astype(int)
            test_df['Time_Order_picked_minute'] = test_df['Time_Order_picked'].dt.minute.astype(int)
            
            logging.info('Completed Separating the hours and minutes into new columns.')
            
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformation_object()

            target_column_name = 'Time_taken (min)'
            drop_columns=[target_column_name,'ID','Delivery_person_ID','Order_Date','Time_Orderd','Time_Order_picked','Type_of_order']

            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name]


            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            ## Transforming using preprocessor object
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            logging.info('Preprocessor pickle file saved')
            
            # # Store the data in MongoDB
            # mongodb_handler = MongoDBHandler()
            # mongodb_handler.insert_data_into_collection(train_arr, test_arr)
            # logging.info('Processed data stored in mongodb')
            
            # Storing the preprocessor object as pkl in mongodb
            mongodb_handler = MongoDBHandler()
            mongodb_handler.save_model_to_mongodb(preprocessing_obj, 'preprocessor')
            


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



            
            
            
        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)