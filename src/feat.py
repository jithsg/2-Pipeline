import pandas as pd
from typing import Text
import yaml
import argparse
def feat(config_path:Text)->None:
    with open(config_path) as conf_file:
        config= yaml.safe_load(conf_file)
    dataset = pd.read_csv(config['data']['processed_path'])
    
    # Create new features
    dataset['sepal_length_to_sepal_width'] = dataset['sepal_length'] / dataset['sepal_width']
    dataset['petal_length_to_petal_width'] = dataset['petal_length'] / dataset['petal_width']
    
    # Select relevant columns
    feature_dataset = dataset[[
        'sepal_length', 'sepal_width', 'petal_length', 'petal_width',
        'sepal_length_to_sepal_width', 'petal_length_to_petal_width',
        'target'
    ]]
    
    feature_dataset.to_csv(config['data']['features_path'], index=False)
    print('Feature engineering complete.')

if __name__ == '__main__':
    argparser= argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
       
    
    feat(config_path=args.config)