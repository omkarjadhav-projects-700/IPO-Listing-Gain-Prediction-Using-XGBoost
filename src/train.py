import xgboost as xgb
from preprocess import preprocess_data
from config import model_path, data_path

X_train, X_test, y_train, y_test = preprocess_data(data_path)

model = xgb.XGBRegressor(
    n_estimators = 200,
    learning_rate = 0.05,
    max_depth = 6,
    subsample = 0.9,
    colsample_bytree = 0.8
)

model.fit(X_train, y_train)
model.save_model(model_path)

print("Model trained and saved successfully!!")