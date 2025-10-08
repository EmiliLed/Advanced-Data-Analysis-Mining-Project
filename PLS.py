
#code functions for Partial Least Squares Regression (PLS)
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#From the data, we need to construct X and Y matrices depending on the hyperparameters number of latent variables and the number of lags

def create_lagged_matrix(data,n_variables, n_lags):
    n_samples = data.shape[0]
    n_features = data.shape[1]
    lagged_data = np.zeros((n_samples - n_lags, n_variables * n_lags))

    for i in range(n_lags, n_samples):
        lagged_row = []
        for j in range(n_lags):
            lagged_row.extend(data[i - j - 1, :n_variables])
        lagged_data[i - n_lags, :] = lagged_row

    return lagged_data


def prepare_data_for_pls(df, target_variable, n_lags):
    # Exclude date column
    data = df.drop(columns=['date']).copy()
    # Convert all columns to numeric, coerce errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')
    # Fill NaN with column means
    data = data.fillna(data.mean())
    data = data.values

    target_index = df.columns.get_loc(target_variable) - 1  # Adjust for dropped date column
    X = create_lagged_matrix(data, data.shape[1], n_lags)
    Y = data[n_lags:, target_index]  # Target variable values aligned with lagged X
    return X, Y

def standardize(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_std = (X - mean) / std
    return X_std, mean, std

def inverse_standardize(X_std, mean, std):
    return X_std * std + mean

def plot_pls_coefficients(pls, df, n_lags):
    # Get variable names (excluding 'date')
    feature_names = [col for col in df.columns if col != 'date']
    # Create lagged feature names
    lagged_feature_names = []
    for lag in range(1, n_lags + 1):
        lagged_feature_names.extend([f"{name}_lag{lag}" for name in feature_names])
    # Plot coefficients
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(pls.coef_.ravel())), pls.coef_.ravel())
    plt.xticks(range(len(lagged_feature_names)), lagged_feature_names, rotation=90)
    plt.ylabel('Regression Coefficient')
    plt.title('PLS Regression Coefficients')
    plt.tight_layout()
    plt.show()

def plot_q2_press_vs_components(df, target_variable, n_lags, max_components=10):
    X, Y = prepare_data_for_pls(df, target_variable, n_lags)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train_std, X_mean, X_std = standardize(X_train)
    X_test_std = (X_test - X_mean) / X_std
    Y_train_std, Y_mean, Y_std = standardize(Y_train.reshape(-1, 1))
    Y_train_std = Y_train_std.ravel()
    Y_test_std = (Y_test.reshape(-1, 1) - Y_mean) / Y_std
    Y_test_std = Y_test_std.ravel()

    q2_list = []
    press_list = []
    for n in range(1, max_components + 1):
        pls = PLSRegression(n_components=n)
        pls.fit(X_train_std, Y_train_std)
        Y_pred_std = pls.predict(X_test_std).ravel()
        Y_pred = inverse_standardize(Y_pred_std, Y_mean, Y_std)
        press = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
        q2 = 1 - press / ss_tot
        q2_list.append(q2)
        press_list.append(press)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, max_components + 1), press_list, marker='o')
    plt.title('PRESS vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('PRESS')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, max_components + 1), q2_list, marker='o')
    plt.title('Q² vs Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Q²')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

def pls_regression(df, target_variable, n_lags=3, n_components=2, test_size=0.2, random_state=42):

    X, Y = prepare_data_for_pls(df, target_variable, n_lags)

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state)

    # Standardize the data
    X_train_std, X_mean, X_std = standardize(X_train)
    X_test_std = (X_test - X_mean) / X_std

    Y_train_std, Y_mean, Y_std = standardize(Y_train.reshape(-1, 1))
    Y_train_std = Y_train_std.ravel()
    Y_test_std = (Y_test.reshape(-1, 1) - Y_mean) / Y_std
    Y_test_std = Y_test_std.ravel()

    # Fit PLS regression
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_train_std, Y_train_std)

    # Predict and inverse standardize
    Y_pred_std = pls.predict(X_test_std).ravel()
    Y_pred = inverse_standardize(Y_pred_std, Y_mean, Y_std)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')

    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.7)
    plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('PLS Regression: Actual vs Predicted')
    plt.grid()
    plt.show()


    return pls, (X_mean, X_std), (Y_mean, Y_std)


#Use with raw data
df = pd.read_csv('MiningProcess_Flotation_Plant_Database.csv', decimal=',')
target_variable = '% Silica Feed'  # Replace with your target variable
pls_model, scaler_X, scaler_Y = pls_regression(df, target_variable, n_lags=3, n_components=20)
plot_pls_coefficients(pls_model, df, n_lags=3)
plot_q2_press_vs_components(df, target_variable, n_lags=3, max_components=20)
pls_model2, scaler_X2, scaler_Y2 = pls_regression(df, target_variable, n_lags=3, n_components=4)
plot_pls_coefficients(pls_model2, df, n_lags=3)
plot_q2_press_vs_components(df, target_variable, n_lags=3, max_components=4)


