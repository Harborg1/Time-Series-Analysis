import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
file_path = "DST_BIL54.csv"
df = pd.read_csv(file_path)

# ---- Inspect column names (uncomment if needed) ----
print(df.head())

# Assume there is a column called 'time' formatted like '2018M01'
# and a column called 'total'

# Convert time column to datetime
df['time'] = pd.to_datetime(df['time'])

# Sort just to be safe
df = df.sort_values('time')

# Create training set (up to 2023-Dec)
train = df[df['time'] <= '2023-12-31'].copy()

# Create time variable x
# x = year + (month-1)/12
train['x'] = train['time'].dt.year + (train['time'].dt.month - 1) / 12

# Extract variable of interest
y = train['total']

# Plot
plt.figure()
plt.plot(train['x'], y)
plt.xlabel('Time (x)')
plt.ylabel('Total registered vehicles')
plt.title('Total Motor Vehicles in Denmark (Training Data)')
plt.show()

# Construct design matrix
X = np.column_stack((np.ones(len(train)), train['x']))

# OLS estimation
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y

theta1_hat = beta_hat[0]
theta2_hat = beta_hat[1]

print("Theta1_hat:", theta1_hat)
print("Theta2_hat:", theta2_hat)
