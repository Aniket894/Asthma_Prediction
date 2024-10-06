# Asthma Diagnosis Prediction ( ML Classification Project ) 

## Asthma Diagnosis Prediction Project Documentation

## Table of Contents
1. Introduction
2. Dataset Description
3. Project Objectives
4. Project Structure
5. Data Ingestion
6. Data Transformation
7. Model Training
8. Training Pipeline
9. Prediction Pipeline
10. Flask (Web Interface)
11. Logging
12. Exception Handling
13. Utils
14. Conclusion

---

## 1. Introduction
The Asthma Diagnosis Prediction project aims to classify patients based on various health and environmental factors to predict asthma diagnosis. This document outlines the project structure, processes, and supporting scripts that drive the model's predictions.

---

## 2. Dataset Description
**Dataset Name:** Asthma Patient Dataset

**Description:**  
The dataset consists of 2,392 entries with 29 columns, capturing a range of health and lifestyle data, used to predict whether a patient has asthma. Below is a brief description of each column:

- `PatientID`: Unique identifier for each patient.
- `Age`: Age of the patient.
- `Gender`: Gender of the patient (encoded).
- `Ethnicity`: Ethnic background (encoded).
- `EducationLevel`: Education level (encoded).
- `BMI`: Body Mass Index.
- `Smoking`: Whether the patient smokes (encoded).
- `PhysicalActivity`: Level of physical activity.
- `DietQuality`: Quality of the patient’s diet.
- `SleepQuality`: Quality of the patient’s sleep.
- `PollutionExposure`: Exposure to pollution.
- `PollenExposure`: Exposure to pollen.
- `DustExposure`: Exposure to dust.
- `PetAllergy`: Whether the patient has pet allergies (encoded).
- `FamilyHistoryAsthma`: Family history of asthma (encoded).
- `HistoryOfAllergies`: General allergy history (encoded).
- `Eczema`: Whether the patient has eczema (encoded).
- `HayFever`: Whether the patient has hay fever (encoded).
- `GastroesophagealReflux`: History of GERD (encoded).
- `LungFunctionFEV1`: Forced Expiratory Volume in one second (FEV1).
- `LungFunctionFVC`: Forced Vital Capacity (FVC).
- `Wheezing`: Whether the patient experiences wheezing (encoded).
- `ShortnessOfBreath`: Whether the patient experiences shortness of breath (encoded).
- `ChestTightness`: Whether the patient experiences chest tightness (encoded).
- `Coughing`: Whether the patient experiences coughing (encoded).
- `NighttimeSymptoms`: Whether the patient has nighttime symptoms (encoded).
- `ExerciseInduced`: Whether asthma symptoms are triggered by exercise (encoded).
- `Diagnosis`: The target column (whether the patient has asthma).
- `DoctorInCharge`: Doctor assigned to the patient (categorical).

---

## 3. Project Objectives
- **Data Ingestion:** Load and explore the asthma patient dataset.
- **Data Transformation:** Clean, preprocess, and transform the dataset for model analysis.
- **Model Training:** Train machine learning models to predict asthma diagnosis.
- **Pipeline Creation:** Develop pipelines for data ingestion, transformation, and model training.
- **Supporting Scripts:** Provide scripts for setup, logging, exception handling, and utility functions.

---

## 4. Project Structure
```
├── artifacts/
│   ├── (best)model.pkl
│   ├── LogisticRegression.pkl
│   ├── RIdgeClassifier.pkl
│   ├── DecisionTreeClassifier.pkl
│   ├── RandomForestClassifier.pkl
│   ├── AdaBoostClassifier.pkl
│   ├── GradientBoostingClassifier.pkl
│   ├── XGBoostClassifier.pkl
│   ├── KNeighborsClassifier.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│     ├── data/
│     │    └── asthma_disease_data.csv
│     └── Asthma_Prediction_(_Classification_).ipynb
├── src/
│   ├── __init__.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_training.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
│
├── templates/
│   ├── index.html
│   └── results.html
├── static/
│   ├── asthma.png
│   └── style.css
│
├── app.py
├── .gitignore
├── requirements.txt
├── README.md
└── setup.py
```

---

## 5. Data Ingestion
The data ingestion module loads the asthma patient dataset, splits it into training and testing sets, and saves them for further use. The raw dataset is stored in the `artifacts/` folder.

---

## 6. Data Transformation
The data transformation module preprocesses the dataset by encoding categorical variables (e.g., `Gender`, `FamilyHistoryAsthma`, `Diagnosis`) and scaling numerical features (e.g., `LungFunctionFEV1`, `PollutionExposure`). The transformed data is stored in the `artifacts/` folder.

---

## 7. Model Training
The model training module trains multiple machine learning classification models, such as:
- Logistic Regression
- Ridge Classifier
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- Gradient Boosting Classifier
- KNeighbors Classifier
- XGB Classifier
  
  
![model_accuracy_comparison](https://github.com/user-attachments/assets/ba80455f-7349-4a10-9143-3c3232ec1d86)



The best-performing model is saved as `best_model.pkl` in the `artifacts/` folder.

---

## 8. Training Pipeline
The training pipeline integrates data ingestion, transformation, and model training, ensuring smooth execution of the workflow from loading data to saving the trained model.

---

## 9. Prediction Pipeline
The prediction pipeline uses `best_model.pkl` and `preprocessor.pkl` to predict asthma diagnosis on new patient data. It handles preprocessing and model inference seamlessly.

---

## 10. Flask (Web Interface)
The Flask app provides a web interface where healthcare professionals can input patient data and predict asthma diagnosis. Input fields are handled by `index.html`, and results are displayed in `results.html`.



![Screenshot 10-06-2024 10 02 22](https://github.com/user-attachments/assets/5c62ce19-a644-42a2-99e2-068c75f0a7ef)


![Screenshot 10-06-2024 10 02 00](https://github.com/user-attachments/assets/e759aabe-6530-4426-b202-427f8a829295)

---

## 11. Logging
The `logger.py` file captures logs for various processes such as data ingestion, transformation, and model training, helping to debug and monitor the workflow.

---

## 12. Exception Handling
The `exception.py` file ensures robust error handling by logging and addressing any issues encountered during the project execution.

---

## 13. Utils
The `utils.py` file includes utility functions for tasks like directory creation, file management, and data loading.

---

## 14. Conclusion
This documentation outlines the end-to-end workflow of the Asthma Diagnosis Prediction project, covering ingestion, transformation, modeling, and deployment. The project is structured to be modular and scalable, facilitating future extensions or adaptations.
