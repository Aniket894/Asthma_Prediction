# src/components/data_transformation.py
import pandas as pd
import numpy as np
import os
import joblib
from src.logger import logging
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def transform_data():
    logging.info("Started data transformation")
    
    # Load train and test datasets
    train_data = pd.read_csv('artifacts/train.csv')
    test_data = pd.read_csv('artifacts/test.csv')
    logging.info("Loaded train and test data")

    # Separate features and target variable
    target = 'Diagnosis'
    cols_to_drop = ['PatientID','DoctorInCharge']
    X_train = train_data.drop(columns=[target]+ cols_to_drop)
    y_train = train_data[target]
    X_test = test_data.drop(columns=[target]+ cols_to_drop)
    y_test = test_data[target]

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    # Define transformations for numerical and categorical columns
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ordinal_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # Combine transformers into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Fit and transform the data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    logging.info("Transformed train and test data")

    # Save the transformed datasets and preprocessor
    np.save('artifacts/X_train.npy', X_train_transformed)
    np.save('artifacts/y_train.npy', y_train)
    np.save('artifacts/X_test.npy', X_test_transformed)
    np.save('artifacts/y_test.npy', y_test)
    joblib.dump(preprocessor, 'artifacts/preprocessor.pkl')
    logging.info("Saved transformed datasets and preprocessor to artifacts folder")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    transform_data()
