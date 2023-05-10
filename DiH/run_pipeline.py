import pandas as pd
from preprocessor import Preprocessor
from model import Model
from sklearn.metrics import roc_curve
import numpy as np
import joblib
import json


class Pipeline:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.threshold = None

    def run(self, data_path, inference=False, output_path=None):
        data = pd.read_csv(data_path)
        cols_to_drop = ['Unnamed: 0', 'In-hospital_death']  # List of column names to check
        X = None
        y = None
        for col in cols_to_drop:
            if col in data.columns:
                if col == 'In-hospital_death':
                    X = data.drop([col], axis=1)
                    y = data[col]
                else:
                    X = data.drop([col], axis=1)
                    y = pd.Series()

        if X is None and y is None:
            X = data
            y = pd.Series()

        if not inference:
            self.preprocessor = Preprocessor()
            self.preprocessor.fit(X)  # Fit the preprocessor to the training data
            X = self.preprocessor.transform(X)  # Transform the training data

            self.model = Model()
            self.model.fit(X, y)

            self.threshold = self.compute_threshold(y, self.model.predict_proba(X)[:, 1])
            joblib.dump(self.preprocessor, 'preprocessor.joblib')
            joblib.dump(self.model, 'model.joblib')
            joblib.dump(self.threshold, 'threshold.joblib')

            print('Training complete')
        else:
            self.preprocessor = joblib.load('preprocessor.joblib')
            self.model = joblib.load('model.joblib')
            self.threshold = joblib.load('threshold.joblib')

            X = self.preprocessor.transform(X)  # Transform the test data
            probabilities = self.model.predict_proba(X)[:, 1]
            outputs = (probabilities >= self.threshold).astype(int)

            data_dict = {'probabilities': probabilities.tolist(),
                         'threshold': float(self.threshold),
                         'predictions': outputs.tolist()}

            with open(output_path, 'w') as f:
                json.dump(data_dict, f)

            print('Predictions saved to', output_path)


    def compute_threshold(self, y_true, probabilities):
        fpr, tpr, thresholds = roc_curve(y_true, probabilities)
        gmeans = np.sqrt(tpr * (1 - fpr))
        ix = np.argmax(gmeans)
        best_threshold = thresholds[ix]
        return best_threshold