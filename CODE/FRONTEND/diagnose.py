import joblib
import pandas as pd
import numpy as np
import os

print("=" * 60)
print("DIAGNOSTIC CHECK FOR SIADD SYSTEM")
print("=" * 60)

# Check 1: Required files
print("\n1. CHECKING REQUIRED FILES:")
required_files = [
    'catboost_model.pkl',
    'scaler.pkl', 
    'top_features.pkl',
    'data/KDDTrain+.txt'
]

for file in required_files:
    if os.path.exists(file):
        print(f"✓ {file} - FOUND")
    else:
        print(f"✗ {file} - NOT FOUND")

# Check 2: Model loading
print("\n2. CHECKING MODEL LOADING:")
try:
    model = joblib.load('catboost_model.pkl')
    print("✓ Model loaded successfully")
    print(f"  Model type: {type(model)}")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

# Check 3: Preprocessors
print("\n3. CHECKING PREPROCESSORS:")
try:
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('top_features.pkl')
    print(f"✓ Scaler loaded: {scaler}")
    print(f"✓ Features loaded: {len(features)} features")
    print(f"  Sample features: {features[:5]}")
except Exception as e:
    print(f"✗ Preprocessor loading failed: {e}")

# Check 4: Dataset
print("\n4. CHECKING DATASET:")
try:
    # Load column names
    column_names = [ 
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
        "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count",
        "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
        "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
        "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack", "difficulty"
    ]
    
    df = pd.read_csv("data/KDDTrain+.txt", names=column_names)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"  First 5 attack types: {df['attack'].unique()[:5]}")
except Exception as e:
    print(f"✗ Dataset loading failed: {e}")

# Check 5: Test prediction
print("\n5. TESTING PREDICTION:")
try:
    if all(os.path.exists(f) for f in ['catboost_model.pkl', 'scaler.pkl', 'top_features.pkl']):
        model = joblib.load('catboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('top_features.pkl')
        
        # Create test input
        test_input = {feature: 0.0 for feature in features}
        test_df = pd.DataFrame([test_input])
        
        # Scale and predict
        scaled_input = scaler.transform(test_df)
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)
        
        print("✓ Prediction successful!")
        print(f"  Predicted class: {prediction[0]}")
        print(f"  Probabilities shape: {probability.shape}")
    else:
        print("✗ Missing files for prediction test")
except Exception as e:
    print(f"✗ Prediction test failed: {e}")

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)