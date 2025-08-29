import subprocess
import os
from drift_detector import detect_raw_drift
from titanic_preprocess import run_preprocessing
from train import run_training

def run_pipeline(file_path):
    """
    Orchestrates the data preprocessing and model training pipeline.
    """
    try:
        print("*** Initiating Automated Retraining Pipeline ***")
        
        # 1. Run data preprocessing script
        print("1.Running Data Preprocessing")
        run_preprocessing(file_path)
        # subprocess.run(["python", "src/titanic_preprocess.py"], check=True)
        print("Data Processing Completed")
        
        # 2. Run Traing 
        print("Running Model Training")
        run_training()
        # subprocess.run(["python", "src/train.py"], check=True)
        print("Model Training & Deployment Completed")
        
        print("*** AUtomated Retraining Pipeline finished Successfully ***")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error: A pipeline step failed. Exiting Retraining. Error: {e}")
        return False
    
    except Exception as e:
        print(f"An UNexpected error occured: {e}")
        return False
    
if __name__ == "__main__":
    reference_data_path = "data/raw/train.csv"
    new_data_path = "data/raw/simulated_drift_data.csv"
    
    drift_detected = detect_raw_drift(reference_data_path, new_data_path)
    
    if drift_detected:
        print("!!! ALERT! Drift Detected. Retraining Pipeline will now run.")
        run_pipeline(file_path = new_data_path)
    else:
        print("No Drift Detected. Retraining is not necessary")