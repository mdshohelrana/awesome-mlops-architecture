# Created by MLflow Pipeliens
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder()

class DataCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
        
    def fit(self, data, y: None):
        return data

    def transform(self, data):
        # Convert 'Datetime' to datetime object and set it as the index
        data['Datetime'] = pd.to_datetime(data['Datetime'])
        data = data.set_index('Datetime')

        data = data.drop(columns=['Timestamp', 'Gmtoffset'])
        return data

# Function to return the unfitted transformer
def transformer_fn():
    return DataCleaner()
