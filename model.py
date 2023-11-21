import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the dataset
file_path = r'D:\Repos\SLM_built_316L_Stainless_Steel_ Property_Prediction\doc\Project.xlsx'
df = pd.read_excel(file_path)

# Loading the data from the correct sheets
train_df = pd.read_excel(file_path, sheet_name='Training Data')
test_df = pd.read_excel(file_path, sheet_name='Test Data')

# Selecting the features and target variables for both train and test sets
features_train = train_df[['Power (W)', 'Scan Speed (mm/s)', 'Hatch spacing (mm)', 'Layer Thickness (mm)']]
targets_train = train_df[['Surface Roughness (μm)', 'Relative Density (%)', 'UTS (MPa)', 'Elongation (%)', 'Hardness (HRB)']]

features_test = test_df[['Power (W)', 'Scan Speed (mm/s)', 'Hatch spacing (mm)', 'Layer Thickness (mm)']]
targets_test = test_df[['Surface Roughness (μm)', 'Relative Density (%)', 'UTS (MPa)', 'Elongation (%)', 'Hardness (HRB)']]

# Data preprocessing
# Convert targets to numeric values, as they contain strings with standard deviations
for df in [targets_train, targets_test]:
    for column in df.columns:
        df[column] = df[column].str.split('±').str[0].astype(float)

# Normalizing the features
scaler = StandardScaler()
features_train_scaled = scaler.fit_transform(features_train)
features_test_scaled = scaler.transform(features_test)

# Splitting the dataset into training and testing sets

# Model Building
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
model.fit(features_train_scaled, targets_train)

# Predicting on test data
y_pred = model.predict(features_test_scaled)

# Model Evaluation
mse = mean_squared_error(targets_test, y_pred)

# Code for saving the model
from joblib import dump
dump(model, 'random_forest_model.joblib')

# Note: The targets splitting and conversion might need adjustments based on the actual data format in the 'Project.xlsx' file.
# The accuracy of the model can be improved by tuning the hyperparameters of the RandomForestRegressor.
# The model can be saved using joblib for later use. Uncomment the last two lines for saving the model.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to evaluate the model
def evaluate_model(model, features_test_scaled, y_test):
    # Making predictions on the test set
    y_pred = model.predict(features_test_scaled)

    # Calculating evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Printing the evaluation metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")

# Example usage
evaluate_model(model, features_test_scaled, targets_test)

