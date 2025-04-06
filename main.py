from src.data_ingestion import DataIngestion
from src.feature_extraction import AudioFeatureExtractor
from src.model_trainer import ModelTrainer
from src.inference import InferenceEngine

if __name__ == "__main__":
    #ingestion = DataIngestion(dataset_dir="dataset")
    #train_df, test_df = ingestion.load_metadata()

    #extractor = AudioFeatureExtractor(sr=16000, n_mfcc=13)
    #train_features = extractor.extract_features(train_df)
    #test_features = extractor.extract_features(test_df)

    #train_features.to_csv("train_features.csv", index=False)
    #test_features.to_csv("test_features.csv", index=False)

    #trainer = ModelTrainer()
    #trainer.train("train_features.csv")

    infer = InferenceEngine()
    infer.predict(
        test_csv="test_features.csv",
        output_csv="submission.csv",
        sample_submission_csv="dataset/sample_submission.csv"
    )



