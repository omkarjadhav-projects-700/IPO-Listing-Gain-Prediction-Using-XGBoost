import logging
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
import xgboost as xgb
from config import model_path, data_path
from preprocess import preprocess_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def evaluate_model():
    logging.info("Loading data...")
    X_train, X_test, y_train, y_test = preprocess_data(data_path)

    logging.info("Loading model...")
    model = xgb.XGBRegressor()
    model.load_model(model_path)

    logging.info("Performing predictions...")
    y_pred = model.predict(X_test)

    # ----- Compute metrics -----
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"RMSE    : {rmse:.4f}")
    logging.info(f"MAE     : {mae:.4f}")
    logging.info(f"R2 Score: {r2:.4f}")

    return rmse, mae, r2
