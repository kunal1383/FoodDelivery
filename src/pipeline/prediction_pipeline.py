import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from src.mongodb import MongoDBHandler
from src.pipeline.training_pipeline import TrainingPipeline

class PredictionPipeline:
    def __init__(self):
        self.training_pipeline = TrainingPipeline()

    def fetch_models(self):
        logging.info('Fetching models from MongoDB')
        mongodb_handler = MongoDBHandler()
        model = mongodb_handler.load_model_from_mongodb('best_model')
        preprocessor = mongodb_handler.load_model_from_mongodb('preprocessor')
        return model, preprocessor

    def load_models_from_artifacts(self):
        preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        model_path = os.path.join('artifacts', 'model.pkl')

        preprocessor = None
        model = None

        try:
            logging.info("Ckecking if models presnt in artifacts")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
        except FileNotFoundError:
            pass

        return model, preprocessor

    def predict(self, features):
        logging.info('Starting the prediction pipeline')
        try:
            model, preprocessor = self.fetch_models()

            if model is not None and preprocessor is not None:
                logging.info('Models fetched successfully from MongoDB')
                data_scaled = preprocessor.transform(features)
                logging.info('Predicting the result')
                pred = model.predict(data_scaled)
                return pred
            else:
                logging.info('Failed to fetch models from MongoDB. Trying to fetch from artifacts.')
                model, preprocessor = self.load_models_from_artifacts()

                if model is not None and preprocessor is not None:
                    logging.info('Models fetched successfully from artifacts')
                    data_scaled = preprocessor.transform(features)
                    logging.info('Predicting the result')
                    pred = model.predict(data_scaled)
                    return pred
                else:
                    logging.info('Models not found in artifacts. Proceeding with model training.')
                    self.training_pipeline.initiate_training()
                    logging.info('Model training completed.')
                    model, preprocessor = self.fetch_models()

                    if model is not None and preprocessor is not None:
                        logging.info('Models fetched successfully from MongoDB after training.')
                        data_scaled = preprocessor.transform(features)
                        logging.info('Predicting the result')
                        pred = model.predict(data_scaled)
                        return pred
                    else:
                        raise CustomException('Models not found after training. Something went wrong.')

        except Exception as e:
            logging.info("Exception occurred in prediction")
            raise CustomException(e, sys)
        
        
