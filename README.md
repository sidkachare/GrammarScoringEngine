
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




