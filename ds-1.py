import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
# Step 1: Generate synthetic data
np.random.seed(42)
brain_size = np.random.normal(1000, 100, 100)  # Brain sizes in cubic cm
iq_scores = 80 + 0.05 * brain_size + np.random.normal(0, 5, 100)  # IQ formula with noise
gender = np.random.choice(['Male', 'Female'], 100)  # Random gender assignment

# Step 2: Display data in tabular form
data = pd.DataFrame({'Brain Size (cm³)': brain_size, 'IQ Score': iq_scores, 'Gender': gender})
print(data.head())

# Step 3: Reshape data for sklearn
X = brain_size.reshape(-1, 1)
y = iq_scores

# Step 4: Train Linear Regression Model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Step 5: Model Performance Evaluation
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Step 6: Visualization
plt.figure(figsize=(8, 5))
plt.scatter(brain_size, iq_scores, color='blue', label="Actual Data")
plt.plot(brain_size, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel("Brain Size (cm³)")
plt.ylabel("IQ Score")
plt.title("Brain Size vs. IQ Score (Linear Regression)")
plt.legend()
plt.show()

# Step 2: Calculate statistical measures for Brain Size and IQ Score

# For Brain Size
brain_size_mean = np.mean(brain_size)
brain_size_median = np.median(brain_size)
brain_size_mode_result = stats.mode(brain_size)  # Get the mode result
brain_size_mode = brain_size_mode_result.mode[0] if brain_size_mode_result.count[0] > 0 else None  # Access mode if present
brain_size_range = np.ptp(brain_size)
brain_size_variance = np.var(brain_size)
brain_size_std_dev = np.std(brain_size)

# For IQ Scores
iq_mean = np.mean(iq_scores)
iq_median = np.median(iq_scores)
iq_mode_result = stats.mode(iq_scores)  # Get the mode result
iq_mode = iq_mode_result.mode[0] if iq_mode_result.count[0] > 0 else None  # Access mode if present
iq_range = np.ptp(iq_scores)
iq_variance = np.var(iq_scores)
iq_std_dev = np.std(iq_scores)

# Step 3: Display the statistical measures
print("Brain Size Statistics:")
print(f"Mean: {brain_size_mean:.2f}")
print(f"Median: {brain_size_median:.2f}")
print(f"Mode: {brain_size_mode:.2f}")
print(f"Range: {brain_size_range:.2f}")
print(f"Variance: {brain_size_variance:.2f}")
print(f"Standard Deviation: {brain_size_std_dev:.2f}")

print("\nIQ Score Statistics:")
print(f"Mean: {iq_mean:.2f}")
print(f"Median: {iq_median:.2f}")
print(f"Mode: {iq_mode:.2f}")
print(f"Range: {iq_range:.2f}")
print(f"Variance: {iq_variance:.2f}")
print(f"Standard Deviation: {iq_std_dev:.2f}")

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")
