from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_training import ModelTraining

def training_pipeline(input_file):
    # Data Ingestion
    logging.basicConfig(level=logging.INFO)
    ingest_data('notebook/data/asthma_disease_data.csv')
    
    # Data Transformation
    logging.basicConfig(level=logging.INFO)
    transform_data()

    
    # Model Training
    model_training = ModelTraining(train_features, train_target, test_features, test_target)
    model_training.train_and_evaluate_models()

if __name__ == "__main__":
    training_pipeline('notebook/data/asthma_disease_data.csv')
