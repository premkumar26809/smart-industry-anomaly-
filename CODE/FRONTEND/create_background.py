import numpy as np
import joblib

# Create realistic background data for SHAP
np.random.seed(42)
n_samples = 100
n_features = 10  # Adjust based on your top_features.pkl

# Create background data with realistic ranges
background_data = np.zeros((n_samples, n_features))

# Column 0-2: Rate features (0-1)
background_data[:, 0:3] = np.random.uniform(0, 1, (n_samples, 3))

# Column 3-7: Count features (0-1000)
background_data[:, 3:8] = np.random.exponential(10, (n_samples, 5))

# Column 8-9: Binary features (0 or 1)
background_data[:, 8:10] = np.random.randint(0, 2, (n_samples, 2))

print(f"Created background data shape: {background_data.shape}")
print(f"Background data stats - Min: {background_data.min()}, Max: {background_data.max()}, Mean: {background_data.mean()}")

# Save to file
joblib.dump(background_data, 'background_data.pkl')
print("Background data saved to background_data.pkl")