
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Text
import yaml
import argparse
def load(config_path:Text)->None:

    with open(config_path) as conf_file:
        config= yaml.safe_load(conf_file)
    # Prepare the dataset
    dataset = pd.read_csv(config['data']['dataset_csv'])
    dataset = pd.DataFrame(dataset)
    dataset = dataset[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "Species"]]
    dataset.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    
    # Encode target labels with value between 0 and n_classes-1.
    le = LabelEncoder()
    dataset.target = le.fit_transform(dataset['target'])
    
    # Save the processed dataset to a new CSV file
    dataset.to_csv(config['data']['processed_path'], index=False)

    print('Data loading complete.')
    
if __name__ == '__main__':
    argparser= argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
       
    
    load(config_path=args.config)