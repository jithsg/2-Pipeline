import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
import argparse
from typing import Text
import yaml

def train(config_path: Text) -> None:
    model_path = 'model/model.pkl'
    file_path = 'data/train_iris.csv'

    # Read the dataset
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    # Load dataset
    dataset = pd.read_csv(file_path)
    y_train = dataset.loc[:, 'target'].values.astype('int32')
    X_train = dataset.drop('target', axis=1).values.astype('float32')

    # Define the parameter grid
    param_grid = {
        'C': config['training']['clf_params']['C'],  # Ensure this is a list
        'solver': config['training']['clf_params']['solver'],  # Ensure this is a list
        'multi_class': config['training']['clf_params']['multi_class'],  # Ensure this is a list
        'max_iter': config['training']['clf_params']['max_iter']  # Ensure this is a list
    }

    # Initialize the logistic regression model
    logreg = LogisticRegression()

    # Perform grid search
    grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Save the best model
    joblib.dump(best_model, model_path)
    print('Training complete. Best params:', grid_search.best_params_)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()

    train(config_path=args.config)
