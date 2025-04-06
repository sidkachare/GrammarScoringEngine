# src/predict_test.py

import pandas as pd
import joblib

def predict(model_path="models/random_forest_regressor.pkl", test_path="test_features.csv", output_path="submission.csv"):
    df = pd.read_csv(test_path)

    X_test = df.drop(columns=["audio_path"])
    
    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    submission = pd.DataFrame({
        "filename": df["audio_path"].apply(lambda x: x.split("/")[-1] if "/" in x else x.split("\\")[-1]),
        "label": predictions
    })

    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")

if __name__ == "__main__":
    predict()
