# src/components/model_training.py
from src.logger import logging
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.stats import uniform, randint
import pickle
import numpy as np

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

def train_models():
    logging.info("Started model training")
    
    # Load transformed datasets
    X_train = np.load('artifacts/X_train.npy',allow_pickle=True)
    y_train = np.load('artifacts/y_train.npy',allow_pickle=True)
    X_test = np.load('artifacts/X_test.npy',allow_pickle=True)
    y_test = np.load('artifacts/y_test.npy',allow_pickle=True)
    logging.info("Loaded transformed datasets")

    models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': uniform(0.001, 100),
                'solver': ['liblinear', 'saga']
            }
        },
        'RidgeClassifier': {
            'model': RidgeClassifier(random_state=42),
            'params': {
                'alpha': uniform(0.01, 100),
                'solver': ['auto', 'sparse_cg', 'lsqr']
            }
        },
        'DecisionTreeClassifier': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {
                'max_depth': [None, 3, 5, 7, 10],
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5),
                'criterion': ['gini', 'entropy']
            }
        },
        'RandomForestClassifier': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 201),
                'max_depth': [None, 3, 5, 7, 10],
                'min_samples_split': randint(2, 11),
                'min_samples_leaf': randint(1, 5),
                'criterion': ['gini', 'entropy']
            }
        },
        'AdaBoostClassifier': {
            'model': AdaBoostClassifier(random_state=42,algorithm='SAMME'),
            'params': {
                'n_estimators': randint(50, 201),
                'learning_rate': uniform(0.01, 1.0)
            }
        },
        'GradientBoostingClassifier': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': randint(50, 201),
                'learning_rate': uniform(0.01, 0.2),
                'max_depth': randint(3, 8),
                'subsample': uniform(0.5, 0.5)
            }
        },
        
        'KNeighborsClassifier': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': randint(3, 10),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        },
        'XGBClassifier': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False),
            'params': {
                'n_estimators': randint(50, 201),
                'learning_rate': uniform(0.01, 0.3),
                'max_depth': randint(3, 11),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5),
                'gamma': uniform(0, 1)
            }
        }
    }

    # Function to evaluate a classification model
    def evaluate_model(model, params, X_train, y_train, X_test, y_test):
        # Perform randomized search
        grid_search = RandomizedSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
        grid_search.fit(X_train, y_train)

        # Get the best model and predictions
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # Calculate accuracy and classification report
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"Model: {best_model}")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Accuracy Score: {accuracy}")
        print("Classification Report:")
        print(report)
        print("=======================================")

        return best_model, accuracy

    adjusted_accuracies = {}
    all_models = {}

    for model_name, model_info in models.items():
        print(f"Evaluating and Training {model_name}...")
        best_model, accuracy = evaluate_model(model_info['model'], model_info['params'], X_train, y_train, X_test, y_test)
        adjusted_accuracies[model_name] = accuracy
        all_models[model_name] = best_model

        # Save the model
        with open(f"artifacts/{model_name}.pkl", 'wb') as file:
            pickle.dump(best_model, file)
        print(f"Model {model_name} saved to artifacts/{model_name}.pkl")

    # Plotting the accuracy comparison
    model_names = list(adjusted_accuracies.keys())
    accuracies = list(adjusted_accuracies.values())

    # Generate a list of colors for each bar
    colors = plt.cm.get_cmap('tab20', len(model_names))

    # Create a bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, accuracies, color=[colors(i) for i in range(len(model_names))])

    # Add labels and title
    plt.xlabel('Model Names')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45, ha='right')

    # Show accuracy values on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.2f}', ha='center', va='bottom')

    # Display the chart
    plt.tight_layout()
    plt.savefig('artifacts/model_accuracy_comparison.png')
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_models()
