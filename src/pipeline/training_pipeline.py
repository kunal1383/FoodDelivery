from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.mongodb import MongoDBHandler

class TrainingPipeline:
    def __init__(self):
        self.train_arr = None
        self.test_arr = None

    def initiate_training(self):
        logging.info("Starting Training Pipeline")
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        self.train_arr, self.test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        
        #mongodb_handler = MongoDBHandler()
        # train_data, test_data = mongodb_handler.fetch_data_from_collection()

        # if train_data is not None and test_data is not None:
        #     # Data successfully fetched from MongoDB
        #     train_df = pd.DataFrame(train_data)
        #     test_df = pd.DataFrame(test_data)
        #     self.train_arr = np.array(train_df)  
        #     self.test_arr = np.array(test_df)    
        #     logging.info("Fetched data from MongoDB.")
        # else:
        #     logging.info("Data not found in MongoDB. Using data from data transformation.")

        model_trainer = ModelTrainer()
        model_trainer.initiate_model_training(self.train_arr, self.test_arr)
        logging.info("Training Pipeline completed successfully")
