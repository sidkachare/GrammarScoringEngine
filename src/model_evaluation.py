import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model_path="models/random_forest_regressor.pkl", data_path="train_features.csv"):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["audio_path", "score"])
    y = df["score"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(model_path)

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    r2 = r2_score(y_val, y_pred)

    print(f"Evaluation Results:")
    print(f" - MAE  : {mae:.4f}")
    print(f" - RMSE : {rmse:.4f}")
    print(f" - R²   : {r2:.4f}")

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_val, y=y_pred)
    plt.xlabel("Actual Grammar Score")
    plt.ylabel("Predicted Grammar Score")
    plt.title("Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        features = X.columns
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [features[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    evaluate()
