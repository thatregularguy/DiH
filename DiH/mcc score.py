from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import roc_auc_score
import pandas as pd
import json

# Load thresholded predictions from json file
with open('predictions.json', 'r') as f:
    predictions_dict = json.load(f)

predictions = predictions_dict['predictions']

# Load true outputs from hospital_deaths_test.csv file
test_df = pd.read_csv('hospital_deaths_test.csv')
y_true = test_df['In-hospital_death']

# Calculate MCC score
auc_score = roc_auc_score(y_true, predictions)
mcc_score = matthews_corrcoef(y_true, predictions)
print(f'AUC score: {auc_score:.3f}')
print(f'MCC score: {mcc_score:.3f}')