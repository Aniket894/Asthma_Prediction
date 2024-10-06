# src/components/data_ingestion.py
import os
import pandas as pd
#import logging
from src.logger import logging
from sklearn.model_selection import train_test_split

def ingest_data(file_path):
    logging.info("Started data ingestion")
    
    # Create artifacts directory if it doesn't exist
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')
    
    # Load the dataset
    data = pd.read_csv("notebook/data/asthma_disease_data.csv")
    logging.info(f"Loaded data from {file_path}")

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    logging.info("Split data into train and test sets")

    # Save raw, train, and test datasets to artifacts folder
    data.to_csv('artifacts/raw.csv', index=False)
    train_data.to_csv('artifacts/train.csv', index=False)
    test_data.to_csv('artifacts/test.csv', index=False)
    logging.info("Saved raw, train, and test datasets to artifacts folder")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_data('notebook/data/asthma_disease_data.csv')
