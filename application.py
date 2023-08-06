from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import PredictionPipeline
from src.pipeline.training_pipeline import TrainingPipeline
from src.utils import parse_date, parse_time
from src.logger import logging
import os
import sys
from src.exception import CustomException 
import pandas as pd 



application = Flask(__name__)
app = application

@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/training', methods=['GET', 'POST'])
def training():
    try:
        training_pipeline = TrainingPipeline()
        training_pipeline.initiate_training()
        render_template("Training Completed")
    except Exception as e:
        raise CustomException(e ,sys)


@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    try:
        if request.method == 'POST':
            
            date_str = request.form.get('Order_Date')
            time_orderd_str = request.form.get('Time_Ordered')
            time_picked_str = request.form.get('Time_Order_picked')
            logging.info(f"Time order:{time_orderd_str} and timeorderPicked:{time_picked_str}")

            year, month, day = parse_date(date_str)
            time_orderd_hour, time_orderd_minute = parse_time(time_orderd_str)
            time_picked_hour, time_picked_minute = parse_time(time_picked_str)
            # Process form data 
            features = {
                'Delivery_person_Age': float(request.form.get('Delivery_person_age')),
                'Delivery_person_Ratings': float(request.form.get('Delivery_person_Ratings')),
                'Restaurant_latitude': float(request.form.get('Restaurant_latitude')),
                'Restaurant_longitude': float(request.form.get('Restaurant_longitude')),
                'Delivery_location_latitude': float(request.form.get('Delivery_location_latitude')),
                'Delivery_location_longitude': float(request.form.get('Delivery_location_longitude')),
                'Vehicle_condition': int(request.form.get('Vehicle_condition')),
                'Road_traffic_density': request.form.get('Road_traffic_density'),
                'Weather_conditions': request.form.get('Weather_conditions'),
                'Type_of_vehicle':request.form.get('Type_of_vehicle'),
                'multiple_deliveries':float(request.form.get('multiple_deliveries')), 
                'Festival':request.form.get('Festival'), 
                'City': request.form.get('City'),
                'year':year, 
                'month':month, 
                'day':day,
                'Time_Orderd_hour':int(time_orderd_hour),
                'Time_Orderd_minute':int(time_orderd_minute), 
                'Time_Order_picked_hour':int(time_picked_hour),
                'Time_Order_picked_minute':int(time_picked_minute)
            }

            # Convert data to DataFrame
            df = pd.DataFrame([features])
            logging.info(df)

            # Predict using the pipeline
            predict_pipeline = PredictionPipeline()
            pred = predict_pipeline.predict(df)

            results = round(pred[0], 2)
            #print(features)
            return render_template('result.html', final_result=results)
            
        
    
    except Exception as e:
        raise CustomException(e, sys)
        return render_template('error.html', error_message=str(e))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000 ,debug=True)
