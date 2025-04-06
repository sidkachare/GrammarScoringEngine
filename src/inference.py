import pandas as pd
import joblib
import os

class InferenceEngine:
    def __init__(self, model_path="models/random_forest_regressor.pkl"):
        self.model = joblib.load(model_path)

    def predict(self, test_csv, output_csv, sample_submission_csv):
        test_df = pd.read_csv(test_csv)

        X_test = test_df.drop(columns=["audio_path"])

        predictions = self.model.predict(X_test)

        submission_format = pd.read_csv(sample_submission_csv)
        audio_file_names = [
            os.path.basename(path) for path in test_df["audio_path"]
        ]

        submission = pd.DataFrame({
            "filename": audio_file_names,
            "label": predictions
        })

        final_submission = submission_format.drop(columns=["label"]).merge(
            submission, on="filename", how="left"
        )

        final_submission.to_csv(output_csv, index=False)
        print(f"Submission file saved to {output_csv}")
