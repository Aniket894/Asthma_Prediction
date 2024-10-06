from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import pandas as pd
from src.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

class PredictionPipeline:
    def __init__(self, model_path, preprocessor_path):
        # Load the pre-trained model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data):
        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data])
        # Transform the data using the preprocessor
        processed_data = self.preprocessor.transform(input_df)
        # Make predictions using the model
        predictions = self.model.predict(processed_data)
        return predictions.tolist()
    
# Initialize the prediction pipeline with model and preprocessor paths
prediction_pipeline = PredictionPipeline(
    model_path='artifacts/GradientBoostingClassifier.pkl',
    preprocessor_path='artifacts/preprocessor.pkl'
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    input_data = {
        'Age': int(request.form.get('Age')),
        'Gender': int(request.form.get('Gender')),
        'Ethnicity': int(request.form.get('Ethnicity')),
        'EducationLevel': int(request.form.get('EducationLevel')),
        'BMI': float(request.form.get('BMI')),
        'Smoking': int(request.form.get('Smoking')),
        'PhysicalActivity': float(request.form.get('PhysicalActivity')),
        'DietQuality': float(request.form.get('DietQuality')),
        'SleepQuality': float(request.form.get('SleepQuality')),
        'PollutionExposure': float(request.form.get('PollutionExposure')),
        'PollenExposure': float(request.form.get('PollenExposure')),
        'DustExposure': float(request.form.get('DustExposure')),
        'PetAllergy': int(request.form.get('PetAllergy')),
        'FamilyHistoryAsthma': int(request.form.get('FamilyHistoryAsthma')),
        'HistoryOfAllergies': int(request.form.get('HistoryOfAllergies')),
        'Eczema': int(request.form.get('Eczema')),
        'HayFever': int(request.form.get('HayFever')),
        'GastroesophagealReflux': int(request.form.get('GastroesophagealReflux')),
        'LungFunctionFEV1': float(request.form.get('LungFunctionFEV1')),
        'LungFunctionFVC': float(request.form.get('LungFunctionFVC')),
        'Wheezing': int(request.form.get('Wheezing')),
        'ShortnessOfBreath': int(request.form.get('ShortnessOfBreath')),
        'ChestTightness': int(request.form.get('ChestTightness')),
        'Coughing': int(request.form.get('Coughing')),
        'NighttimeSymptoms': int(request.form.get('NighttimeSymptoms')),
        'ExerciseInduced': int(request.form.get('ExerciseInduced'))
    }

    # Make prediction
    predictions = prediction_pipeline.predict(input_data)
    
    return render_template('results.html', predictions=predictions)
    
if __name__ == '__main__':
    app.run(debug=True)
