import os
import pandas as pd

class DataIngestion:
    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = dataset_dir
        self.train_csv = os.path.join(dataset_dir, "train.csv")
        self.test_csv = os.path.join(dataset_dir, "test.csv")
        self.audio_train_dir = os.path.join(dataset_dir, "audios_train")
        self.audio_test_dir = os.path.join(dataset_dir, "audios_test")

    def load_metadata(self):
        train_df = pd.read_csv(self.train_csv)
        test_df = pd.read_csv(self.test_csv)

        train_df["audio_path"] = train_df["filename"].apply(
            lambda x: os.path.join(self.audio_train_dir, x)
        )
        test_df["audio_path"] = test_df["filename"].apply(
            lambda x: os.path.join(self.audio_test_dir, x)
        )

        missing_train = train_df[~train_df["audio_path"].apply(os.path.exists)]
        missing_test = test_df[~test_df["audio_path"].apply(os.path.exists)]

        if not missing_train.empty:
            raise FileNotFoundError(f"Missing training files:\n{missing_train}")
        if not missing_test.empty:
            raise FileNotFoundError(f"Missing test files:\n{missing_test}")

        return train_df, test_df
