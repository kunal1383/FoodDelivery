import pymongo
import os
import sys
import pandas as pd
import pickle
from src.exception import CustomException
from src.logger import logging
from src.utils import MONGO_USERNAME, MONGO_PASSWORD, MONGO_DB ,COLLECTION_NAME

class MongoDBHandler:
    def __init__(self):
        self.client = pymongo.MongoClient(f"mongodb+srv://{MONGO_USERNAME}:{MONGO_PASSWORD}@cluster0.thebkor.mongodb.net/")
        self.db = self.client[MONGO_DB]
        
    def insert_data_into_collection(self, train_data, test_data):
        logging.info(f'Connecting to MongoDB')
        logging.info(f'{self.db}')

        try:
            if COLLECTION_NAME in self.db.list_collection_names():
                logging.info(f'Collection: {COLLECTION_NAME}')
                collection = self.db[COLLECTION_NAME]
            else:
                logging.info(f'{COLLECTION_NAME} creating a new collection')
                collection = self.db.create_collection(COLLECTION_NAME)

            logging.info('Inserting into collection')
            # Convert NumPy ndarrays to DataFrames
            train_df = pd.DataFrame(train_data)
            test_df = pd.DataFrame(test_data)

            # Add a 'dataset' column to distinguish between train and test data
            train_df['dataset'] = 'train'
            test_df['dataset'] = 'test'

            # Convert DataFrames to dictionaries
            train_data_dict = train_df.to_dict(orient='records')
            test_data_dict = test_df.to_dict(orient='records')

            # Insert train data
            collection.insert_many(train_data_dict)

            # Insert test data
            collection.insert_many(test_data_dict)

            logging.info(f"Inserted {len(train_data_dict)} records into the collection '{COLLECTION_NAME}' for training data")
            logging.info(f"Inserted {len(test_data_dict)} records into the collection '{COLLECTION_NAME}' for test data")
            logging.info(f"Completed inserting data into MongoDB")

        except Exception as e:
            logging.info(f"An error occurred while inserting data into MongoDB collection")
            raise CustomException(e, sys)


        
        
    def fetch_data_from_collection(self):
        try:
            logging.info(f"Fetching processed data from MongoDB")
            logging.info(f"{self.db}")

            if COLLECTION_NAME in self.db.list_collection_names():
                logging.info(f"Collection: {COLLECTION_NAME}")
                collection = self.db[COLLECTION_NAME]
            else:
                logging.info(f"{COLLECTION_NAME} not found in the database.")
                return None, None

            logging.info("Fetching train data from the collection")
            train_data = list(collection.find({'dataset': 'train'}, {'_id': 0}))

            logging.info("Fetching test data from the collection")
            test_data = list(collection.find({'dataset': 'test'}, {'_id': 0}))

            return train_data, test_data

        except Exception as e:
            logging.info(f"An error occurred while fetching data from MongoDB collection")
            raise CustomException(e, sys)
        
        
    def save_model_to_mongodb(self, model, model_name):
        logging.info(f"Saving model '{model_name}' to MongoDB")

        try:
            if COLLECTION_NAME in self.db.list_collection_names():
                logging.info(f"Collection: {COLLECTION_NAME}")
                collection = self.db[COLLECTION_NAME]
            else:
                logging.info(f"{COLLECTION_NAME} creating a new collection")
                collection = self.db.create_collection(COLLECTION_NAME)

            # Serialize the model to pickle format
            model_pickle = pickle.dumps(model)

            # Create a document to store the model
            model_doc = {"model_name": model_name, "model_pickle": model_pickle}

            # Insert the document into MongoDB
            collection.insert_one(model_doc)

            logging.info(f"Model '{model_name}' saved to MongoDB")

        except Exception as e:
            logging.info(f"An error occurred while saving model '{model_name}' to MongoDB")
            raise CustomException(e, sys)

    def load_model_from_mongodb(self, model_name):
        logging.info(f"Loading model '{model_name}' from MongoDB")

        try:
            if COLLECTION_NAME in self.db.list_collection_names():
                logging.info(f"Collection: {COLLECTION_NAME}")
                collection = self.db[COLLECTION_NAME]
            else:
                logging.info(f"{COLLECTION_NAME} not found in the database.")
                return None

            # Find the model document by model_name
            model_doc = collection.find_one({"model_name": model_name})

            if model_doc is None:
                logging.info(f"Model '{model_name}' not found in MongoDB.")
                return None

            # Deserialize the model from the pickle data
            model_pickle = model_doc["model_pickle"]
            model = pickle.loads(model_pickle)

            logging.info(f"Model '{model_name}' loaded from MongoDB")
            return model

        except Exception as e:
            logging.info(f"An error occurred while loading model '{model_name}' from MongoDB")
            raise CustomException(e, sys)    
