import os 
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging

#Ensure 'log' directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

#logging Configuration
logger = logging.getLogger('model_evalution')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
filer_handler = logging.FileHandler(log_file_path)
filer_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
filer_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(filer_handler)

def load_model(file_path):
    '''
    Load the trained model form the file
    '''
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('MOdel loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logger.debug('File not found %s', file_path)
        raise
    except Exception as e:
        logger.debug('Unexpected error occured while loading the model: %s', e)
        raise


def load_data(file_path):
    '''
    Load data from a CSV file
    '''
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data loaded from: %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Fialed tp parse the CSV File: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data: %s', e)
        raise


def evulate_model(clf,X_test,y_test):
    '''
    Evalute the model and return the evulation metrics
    '''
    try: 
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:,1]
        
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)

        metrics_dict ={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc': auc
        }

        logger.debug('Model Evalution metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evulation: %s', e)
        raise

def save_matrics(metrics,file_path):
    '''
    Save the evulation metrics calculated
    '''
    try:
        #Ensure the directory exists
        os.makedirs(os.path.dirname(file_path),exist_ok=True)

        with open (file_path, 'w') as file:
            json.dump(metrics,file, indent=4)
        logger.debug('Metrics saved to %s', file_path)

    except Exception as e:
        logger.error('Error occured while saving the metrics: %s', e)
        raise



def main():
    try: 
        clf = load_model('/home/amit/Desktop/MLops/MLOPPs_MLpipeline/MLOPs_complete_ML/models/model.pkl')
        test_data = load_data('/home/amit/Desktop/MLops/MLOPPs_MLpipeline/MLOPs_complete_ML/data/processed/test_tfidf.csv')
        
        X_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values

        metrics = evulate_model(clf,X_test,y_test)
        
        save_matrics(metrics, 'reports/metrics.json')
    except Exception as e:
        logger.error('Failed to complete  the model evulation process %s', e)
        print(f'Error: {e}')


if __name__ == '__main__':
    main()
    


