import pandas as pd
import numpy as np

import os

from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score


def save_metrics(model, x_test, y_test, encoder, tag=None, path=r'd:\Documents\ZNU\диплом\metrics.csv'):

    err, acc = model.evaluate(x_test, y_test)

    
    y_pred = model.predict(x_test)
    y_pred_decoded = encoder.inverse_transform(y_pred)
    y_true_decoded = encoder.inverse_transform(y_test)


    if tag == None:
        tag = model.name

    metrics_map = {'model': tag, 'recall': 0,
                   'precision': 0, 'accuracy': 0, 'f1': 0}

    metrics_map['recall'] = recall_score(
        y_true=y_true_decoded, y_pred=y_pred_decoded, average='micro')
    metrics_map['precision'] = precision_score(
        y_true=y_true_decoded, y_pred=y_pred_decoded, average='micro')
    metrics_map['accuracy'] = accuracy_score(
        y_true=y_true_decoded, y_pred=y_pred_decoded)
    metrics_map['f1'] = f1_score(
        y_true=y_true_decoded, y_pred=y_pred_decoded, average='micro')

    results = pd.DataFrame(metrics_map, index=[0])

    if (os.path.exists(path)):
        saved_metrics = pd.read_csv(path)
        saved_metrics = saved_metrics[saved_metrics['model'] != tag]
        results = results.append(saved_metrics)
        
    results.to_csv(path, index=False)
