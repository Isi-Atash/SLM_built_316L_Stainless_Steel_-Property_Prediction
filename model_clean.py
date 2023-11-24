# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to split target values into central and variance parts
def split_target_values(df, column_name):
    y_central = []
    y_variance = []

    for value in df[column_name]:
        if '±' in value:
            central, variance = [float(part.strip()) for part in value.split('±')]
            y_central.append(central)
            y_variance.append(variance)

    return pd.DataFrame(y_central, columns=[f'Central {column_name}']), pd.DataFrame(y_variance, columns=[f'Variance {column_name}'])

# Load the dataset
file_path = r'D:\Repos\SLM_built_316L_Stainless_Steel_ Property_Prediction\doc\Project.xlsx'
df_training = pd.read_excel(file_path, sheet_name='Training Data')
df_test = pd.read_excel(file_path, sheet_name='Test Data')

# Split features and targets
x_train = df_training.iloc[:, 1:5]
y_surface_train, y_variance_train = split_target_values(df_training, 'Surface Roughness (μm)')
y_density_train, y_variance_density = split_target_values(df_training, 'Relative Density (%)')

x_test = df_test.iloc[:, 1:5]
y_surface_test, y_variance_test = split_target_values(df_test, 'Surface Roughness (μm)')
y_density_test, y_variance_density_test = split_target_values(df_test, 'Relative Density (%)')

# Normalize input features
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Define and compile the model
def build_model(input_shape, output_units=1, learning_rate=0.001):
    x_input = Input(shape=(input_shape,), name="x_input")
    x = Dense(32, activation="relu")(x_input)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(output_units)(x)
    model = Model(x_input, output)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model

# Train and evaluate the model
def train_and_evaluate(model, x_train, y_train, x_test, y_test, batch_size, epochs):
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=True)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    plt.scatter(y_test, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    plt.plot([min(y_test)[0], max(y_test)[0]], [min(y_pred)[0], max(y_pred)[0]], linestyle='--', color='red', linewidth=2)
    plt.show()
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (R2): {r2}")


# Model for Surface Roughness
model_surface = build_model(x_train_scaled.shape[1])
train_and_evaluate(model_surface, x_train_scaled, y_surface_train, x_test_scaled, y_surface_test, batch_size=50, epochs=500)

# Model for Relative Density
model_density = build_model(x_train_scaled.shape[1])
train_and_evaluate(model_density, x_train_scaled, y_density_train, x_test_scaled, y_density_test, batch_size=20, epochs=1000)



print("FINISHED")