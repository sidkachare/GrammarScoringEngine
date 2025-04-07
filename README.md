
# Grammar Scoring Engine

This project builds a **regression model** to predict **grammar proficiency scores** (0–5 Likert scale) from audio recordings of human speech. Each audio file is between 45–60 seconds and scored based on grammar quality using a predefined rubric.

---
## Project Structure
    The project structure is modular, having separate files for each module.
    
    GrammarScoringEngine
        --dataset/
            --audios_test
            --audios_train
            --sample_submission.csv
            --test.csv
            --train.csv
        --models/
            --random_forest_regressor.pkl
        --outputs/
            --evaluation_results.txt
            --actual vs. predicted.png
            --feature importance.png
        --src/
            --data_ingestion.py
            --feature_extraction.py
            --inference.py
            --model_evaluation.py
            --model_trainer.py
            --predict.py
        
        --main.py
        --requirements.txt
        --submission.csv
        --test_features.csv
        --train_features.csv


## Instructions

1. Clone the Repository

2. Create and Activate Virtual Environment

3. Install Dependencies:
    pip install -r requirements.txt

4. Run the Pipeline:
    python main.py

5. Run Predictions on Test Set:
    python src/predict.py


## Model
    Model is stored inside models/random_forest_regressor.pkl

## Evaluation
    The evaluation results and visualizations are stored in the outputs folder.


# Project Report

### Dataset Overview:
--Train Set: 444 audio files + train.csv (contains file_name and label)

--Test Set: 195 audio files + test.csv (contains file_name)

--Each audio file is between 45 to 60 seconds long.

### Pipeline Architecture:
The entire pipeline was modularized and built with reusability in mind:

1. Data Ingestion
2. Audio Feature Extraction
3. Model Training
4. Model Evaluation
5. Test Set Prediction


### Preprocessing & Feature Extraction
Each audio file is preprocessed and converted into numeric features for model input. We extract the following features:

--MFCCs (13) – Capture spectral structure

--Zero-Crossing Rate (ZCR) – Voicing and energy content

--Root Mean Square Energy (RMSE) – Loudness

--Spectral Centroid – Brightness of audio

--Pitch – Average pitch across the clip

### Model Training
Model Used: RandomForestRegressor from sklearn.ensemble

Train-Test Split: 80–20 (using train_test_split)

Hyperparameters: Default settings (can be tuned later)

Model trained on train_features.csv where:

Features: 17 extracted audio-based features

Target: label (grammar score)

The trained model was saved as random_forest_regressor.pkl.

### Model Evaluation
Evaluation was performed on the validation split using:
Metric:
--MAE (Mean Absolute Error)
--RMSE (Root Mean Squared Error)
--R² Score

### Conclusion
This project demonstrates a complete pipeline for scoring spoken grammar proficiency using raw audio. It includes audio preprocessing, feature engineering, regression modeling, evaluation, and prediction.

Further enhancements can include:
--Using deep learning or transformer based architectures.
--Text-based grammar scoring (via ASR → NLP).
--Model ensembling.
--Real-time feedback loop for learning improvement.

## Thank You!



