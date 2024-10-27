# Disaster Response Pipeline Project
This project processes disaster-related messages to identify categories that assist in routing messages to appropriate response teams. It uses an ETL pipeline to clean and store data, and a machine learning pipeline to train a model for classifying disaster messages. The web app allows users to enter new messages and receive category predictions to help prioritize disaster response.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ or http://127.0.0.1:3001


