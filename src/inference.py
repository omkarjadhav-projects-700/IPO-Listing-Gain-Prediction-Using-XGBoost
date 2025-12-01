import xgboost as xgb
import numpy as np
import pandas as pd
from config import model_path

class Prediction_Model:
    def __init__(self):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)

    def predict(self, data_dict: dict):
        df = pd.DataFrame([data_dict])
        df = pd.get_dummies(df)
        # Add missing columns as zero
        for col in self.model.get_booster().feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self.model.get_booster().feature_names]
        pred = self.model.predict(df)[0]
        return int(pred)

model_instance = Prediction_Model()