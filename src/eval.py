import yaml
import pandas as pd
import joblib
import json
import csv

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, f1_score,
    precision_score, recall_score, accuracy_score
)
import matplotlib.pyplot as plt
import argparse
from typing import Text

def eval(config_path: Text) -> None:
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
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
    precision = precision_score(y_true=y_test, y_pred=prediction, average='macro')
    recall = recall_score(y_true=y_test, y_pred=prediction, average='macro')
    accuracy = accuracy_score(y_true=y_test, y_pred=prediction)

    # Create a Confusion Matrix Display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save the figure
    plt.savefig(config['reports']['confusion_matrix_file'])

    # Save Confusion Matrix Data to CSV
    cm_csv_file = config['reports']['confusion_matrix_csv_file']
    with open(cm_csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True', 'Predicted'])
        for true_class, predicted_class in zip(y_test, prediction):
            writer.writerow([true_class, predicted_class])

    # Metrics
    metrics = {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy
    }

    with open(config['reports']['metrics_file'], 'w') as mf:
        json.dump(obj=metrics, fp=mf, indent=4)

    print('Evaluation complete.')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--config", dest="config", required=True)
    args = argparser.parse_args()

    eval(config_path=args.config)
