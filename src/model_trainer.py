import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os


class ModelTrainer:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def train(self, features_path: str = "train_features.csv"):
        df = pd.read_csv(features_path)

        X = df.drop(columns=["audio_path", "score"])
        y = df["score"]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred) ** 0.5
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        print(f"Model Performance:")
        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE : {mae:.4f}")
        print(f"   R²  : {r2:.4f}")

        model_path = f"{self.model_dir}/random_forest_regressor.pkl"
        joblib.dump(model, model_path)
        print(f"Model saved to: {model_path}")

        return model, (rmse, mae, r2)
