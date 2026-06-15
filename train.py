import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_model():
    print("Starting AI-Based Queue & Crowd Prediction Model Training...")
    
    # 1. Load data
    data_path = 'crowd_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} records.")
    
    # Ensure correct columns exist
    required_cols = ['hour', 'day_of_week', 'people_count']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing in the dataset.")
            
    # 2. Features and Target
    X = df[['hour', 'day_of_week']]
    y = df['people_count']
    
    # 3. Split into Train & Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Split data into train size ({len(X_train)}) and test size ({len(X_test)}).")
    
    # 4. Train Model
    # Random Forest Regressor fits non-linear relationships well (e.g., peak rush hours)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    model.fit(X_train, y_train)
    
    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Evaluation Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.2f} people")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f} people")
    print(f"  R-squared Score (R2): {r2:.4f}")
    
    # 6. Save Model
    model_filename = 'queue_model.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel successfully saved to '{model_filename}'")
    
    # Save training metadata/metrics for dashboard consumption
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'dataset_size': int(len(df))
    }
    joblib.dump(metrics, 'model_metrics.pkl')
    print("Metrics metadata saved to 'model_metrics.pkl'")

if __name__ == "__main__":
    train_model()
