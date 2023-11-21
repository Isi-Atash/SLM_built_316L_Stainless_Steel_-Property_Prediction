import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.impute import SimpleImputer

# Load the dataset from your local drive
file_path = r'D:\Repos\SLM_built_316L_Stainless_Steel_ Property_Prediction\doc\Project.xlsx'

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
# For each DataFrame in the list
for df in [targets_train, targets_test]:
    # Make sure to iterate over a copy of the DataFrame to avoid SettingWithCopyWarning
    for column in df.columns:
        # Convert the column to numeric, splitting off any additional text
        df.loc[:, column] = df[column].str.split('±').str[0].astype(float)


# Handling Missing Values (if any)
imputer = SimpleImputer(strategy='mean')  # or strategy='median'
features_train = pd.DataFrame(imputer.fit_transform(features_train), columns=features_train.columns)
features_test = pd.DataFrame(imputer.transform(features_test), columns=features_test.columns)

# # Scaling/Normalization
# scaler = StandardScaler()
# features_train_scaled = scaler.fit_transform(features_train)
# features_test_scaled = scaler.transform(features_test)

# After these steps, your data is ready for model training and evaluation.
print((features_train))