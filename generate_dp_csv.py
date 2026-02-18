import pandas as pd
import numpy as np

# Load perturbed.csv
df = pd.read_csv("perturbed.csv")

# Extract features and labels
X = df[['source', 'edge']].values
Y = df['label'].values

# Add Differential Privacy noise
epsilon = 1.0
sensitivity = 1.0
scale = sensitivity / epsilon
noise = np.random.laplace(loc=0.0, scale=scale, size=X.shape)
X_noised = X + noise

# Save as new CSV: perturbed_dp.csv
dp_df = pd.DataFrame(X_noised, columns=['source', 'edge'])
dp_df['label'] = Y
dp_df.to_csv("perturbed_dp.csv", index=False)

print("✅ perturbed_dp.csv created successfully.")
