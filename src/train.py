
import joblib
from sklearn.linear_model import LogisticRegression
import pandas as pd
import argparse
from typing import Text
import yaml
def train(config_path:Text)->None:
    model_path ='model/model.pkl'
    file_path = 'data/train_iris.csv'
    # Read the dataset
    with open(config_path) as conf_file:
        config= yaml.safe_load(conf_file)

    # Load dataset
    dataset = pd.read_csv(file_path)
    y_train = dataset.loc[:, 'target'].values.astype('int32')
    X_train = dataset.drop('target', axis=1).values.astype('float32')

    # Initialize and train the logistic regression model
    logreg = LogisticRegression(
        C=config['training']['clf_params']['C'], 
        solver=config['training']['clf_params']['solver'],
        multi_class=config['training']['clf_params']['multi_class'], 
        max_iter=config['training']['clf_params']['max_iter']
    )
    logreg.fit(X_train, y_train)

    # Save the trained model to the specified path
    joblib.dump(logreg, model_path)
    print('Training complete.')
    # Optionally, return the trained model
    return logreg
    

if __name__ == '__main__':
    argparser= argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
       
    
    train(config_path=args.config)