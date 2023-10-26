import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml
import argparse
def split(config_path:Text)->None:
    # Read the dataset
    with open(config_path) as conf_file:
        config= yaml.safe_load(conf_file)
    dataset = pd.read_csv(config['data']['features_path'])
    
    # Split the dataset into training and testing sets
    train_dataset, test_dataset = train_test_split(dataset, test_size=config['data']['test_size'], random_state=config['data']['random_state'])
    
    # Save the training and testing sets to their respective paths
    train_dataset.to_csv(config['data']['trainset_path'], index=False)
    test_dataset.to_csv(config['data']['testset_path'], index=False)
    
    print('Data splitting complete.')
    
if __name__ == '__main__':
    argparser= argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
       
    
    split(config_path=args.config)