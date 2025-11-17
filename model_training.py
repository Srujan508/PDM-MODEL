import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# --- Configuration ---
MODEL_PATH = 'models/pdm_rf_model.pkl'

def train_and_save_model():
    # --- 1. Load Data ---
    # NOTE: In a real environment, you would load your full dataset here. 
    # For this demonstration, we use the small, combined maintenance log data.
    
    data = {
        'unit_id': ['88-525001', '27-7200', '_OVERLOCK machine', '6-206813', '-TSD'],
        'machine_type': ['KANSAI machine', 'single needele machine', '_OVERLOCK machine', 'BARTACK machine', 'Keyole machine'],
        'weekly_score': [40, 36, 38, 35, 38], # Extracted total weekly checks
        'monthly_score': [9, 9, 8, 8, 9]     # Extracted total monthly checks
    }
    df = pd.DataFrame(data)

    if not os.path.exists('models'):
        os.makedirs('models')

    print("Data loaded. Preparing features and target...")

    # --- 2. Feature Engineering & Target Definition ---
    # Target (y): 1 if the weekly compliance is poor (score < 37), 0 otherwise (Hypothetical risk)
    RISK_THRESHOLD = 37 
    df['Risk_Target'] = df['weekly_score'].apply(lambda x: 1 if x < RISK_THRESHOLD else 0)

    X = df[['weekly_score', 'monthly_score']]
    y = df['Risk_Target']

    # --- 3. Model Training ---
    # NOTE: With only 5 data points, training a model is primarily for structure demonstration.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y) # Training on full small set for stability

    # --- 4. Save the Model and Feature List ---
    joblib.dump({
        'model': model, 
        'features': X.columns.tolist(),
    }, MODEL_PATH)

    print(f"\n✅ Model trained and saved successfully to {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save_model()