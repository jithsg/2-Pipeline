import yaml
from typing import Text
import pandas as pd
import joblib
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt
import argparse
def eval(config_path:Text)->None:
    with open(config_path) as conf_file:
        config= yaml.safe_load(conf_file)
        model = joblib.load(config['training']['model_path'])
    test_dataset = pd.read_csv(config['data']['testset_path'])
    # Prepare test data
    y_test = test_dataset.loc[:, 'target'].values.astype('int32')
    X_test = test_dataset.drop('target', axis=1).values.astype('float32')

    # Make predictions
    prediction = model.predict(X_test)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, prediction)
    f1 = f1_score(y_true=y_test, y_pred=prediction, average='macro')

    # Create a Confusion Matrix Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save the figure
    plt.savefig(config['reports']['confusion_matrix_file'])

    # Show the plot

    metrics_file = config['reports']['metrics_file']

    metrics = {
        'f1': f1
    }

    with open(metrics_file, 'w') as mf:
        json.dump(
            obj=metrics,
            fp=mf,
            indent=4
        )
    print('Evaluation complete.')
    
if __name__ == '__main__':
    argparser= argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()
       
    
    eval(config_path=args.config)