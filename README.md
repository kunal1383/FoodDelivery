
# Food Delivery Prediction



## Introduction

Food Delivery Prediction is a machine learning project that aims to predict the time taken for food delivery based on various factors such as weather conditions, road traffic density, delivery person's ratings, and more. The project utilizes a machine learning model trained on historical delivery data to make accurate predictions.

## Getting Started

To get started with the project, follow the steps below:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/kunal1383/FoodDelivery.git
   ```

2. Create a new virtual environment for the project:

   ```bash
   conda create -p ./venv

   conda activate ./venv

   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Obtain your MongoDB Atlas username and password. These credentials will be used to save the trained machine learning model and later accessed during the prediction pipeline.

## Important Note - MongoDB Credentials

For the Food Delivery Prediction to work correctly, you need to provide your MongoDB Atlas `username and password` in `src/utils.py file`. These credentials are used to store and retrieve the trained machine learning model for predictions.





## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/your-username/FoodDelivery/blob/main/LICENSE) file for details.
