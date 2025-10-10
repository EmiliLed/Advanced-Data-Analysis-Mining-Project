
#code functions for Partial Least Squares Regression (PLS)
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
now = datetime.now()


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
def q2_cal(y_true,y_pred,y_mean_cal):
    press = np.sum((y_true - y_pred) ** 2)
    tss_cal = np.sum((y_true - y_mean_cal) ** 2)
    return 1 - (press / tss_cal)

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

def plot_pls_coefficients_direct(pls, df):
    # Get variable names (excluding 'date')
    feature_names = [col for col in df.columns if col != 'date']
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(pls.coef_.ravel())), pls.coef_.ravel())
    plt.xticks(range(len(feature_names)), feature_names, rotation=90)
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

def pls_regression_adj_cv(X_train, Y_train, max_components=2,initial_train_size=400,test_size=20, step_size=20,number_lags=9,name_y="% Silica Concentrate_mean_lag"):
    activated_pred_window=False
    n_samples = X_train.shape[0]
    q2_array=[]
    r2_array=[]
    mse_array=[]
    X_train_std, X_mean, X_std = standardize(X_train.values)
    Y_train_std, Y_mean, Y_std = standardize(np.asarray(Y_train).reshape(-1, 1))

    if X_train.shape[0] < initial_train_size:
        raise ValueError('Initial training size is too small')
    for start in range(initial_train_size, n_samples - test_size + 1, step_size):
        train_start = start - initial_train_size # alternavily for extended window set 0
        train_end = start
        test_end = start + test_size
        X_train_help = X_train.values[train_start:train_end]
        y_train_help = Y_train.values[train_start:train_end]
        X_validation = X_train.values[train_end:test_end]
        y_validation = Y_train.values[train_end:test_end]
        X_train_std= (X_train_help- X_mean) / X_std
        X_test_std = (X_validation - X_mean) / X_std

        Y_train_std=(y_train_help.reshape(-1, 1) - Y_mean) / Y_std
        Y_train_std = Y_train_std.ravel()
        Y_test_std = (y_validation.reshape(-1, 1) - Y_mean) / Y_std
        Y_test_std = Y_test_std.ravel()

        # Fit PLS regression
        pls = PLSRegression(n_components=max_components)
        pls.fit(X_train_std, Y_train_std)

        # Predict and inverse standardize
        y_pred_array = []
        y_pred_to_x_array = []
        if activated_pred_window:
            for index, x_values in enumerate(X_test_std):
                print(index)
                print(y_pred_to_x_array)
                for i in range(len(y_pred_to_x_array)):
                    x_values[X_train.columns.get_loc(name_y + str(i+1))] = y_pred_to_x_array[i]
                y_pred = pls.predict(x_values.reshape(1, -1)).item()
                y_pred_array.append(y_pred)
                y_pred_to_x_array = [y_pred] + y_pred_to_x_array
                while len(y_pred_to_x_array) > number_lags:
                    print("loop")
                    y_pred_to_x_array = y_pred_to_x_array[:number_lags]
        else:
            y_pred_array = pls.predict(X_test_std).ravel()

        k1 = np.sum((Y_test_std - y_pred_array) ** 2)
        k2 = np.sum((Y_test_std - np.mean(Y_train_std)) ** 2)
        q2 = 1 - k1 / k2
        #Y_pred_std = pls.predict(X_test_std).ravel()
        Y_pred = inverse_standardize(y_pred_array, Y_mean, Y_std)
        q2_array.append([train_start,train_end,test_end,q2])

        # Evaluate the model
        mse = mean_squared_error(y_validation, Y_pred)
        r2 = r2_score(y_validation, Y_pred)
        mse_array.append([train_start,train_end,test_end,mse])
        r2_array.append([train_start,train_end,test_end,r2])

    X_train_std = (X_train.values - X_mean) / X_std
    Y_train_std = (Y_train.values.reshape(-1, 1) - Y_mean) / Y_std
    Y_train_std = Y_train_std.ravel()
    pls = PLSRegression(n_components=max_components)
    pls.fit(X_train_std, Y_train_std)

    return pls, (X_mean, X_std), (Y_mean, Y_std),np.mean(Y_train_std),q2_array,mse_array,r2_array

def testing_timeseries_(x,model,x_columns,number_lags=9,name_y="% Silica Concentrate_mean_lag",simple=True):
    y_pred_array = []
    y_pred_to_x_array = []
    if not simple:
        for index, x_values in enumerate(x):
            print(index)
            print(y_pred_to_x_array)
            for i in range(len(y_pred_to_x_array)):
                x_values[x_columns.get_loc(name_y + str(i+1))] = y_pred_to_x_array[i]
            y_pred = model.predict(x_values.reshape(1, -1)).item()
            y_pred_array.append(y_pred)
            y_pred_to_x_array = [y_pred] + y_pred_to_x_array
            while len(y_pred_to_x_array) > number_lags:
                print("loop")
                y_pred_to_x_array = y_pred_to_x_array[:number_lags]

        return y_pred_array
    else:
        for index, x_values in enumerate(x):
            y_pred = model.predict(x_values.reshape(1, -1)).item()
            y_pred_array.append(y_pred)
        return y_pred_array

def get_lagged_data(df_original_norm,lagging_rule):
    df_lagged = pd.DataFrame(index=df_original_norm.index)
    for index,key in enumerate(df_original_norm.columns):
        for i in range(lagging_rule[index]):
            lag=i+1
            df_lagged[f'{key}_lag{lag}'] = df_original_norm[key].shift(lag)
    df_lagged[f"y_goal_{df_original_norm.keys()[-1]}"] = df_original_norm[df_original_norm.keys()[-1]]
    df_lagged.dropna(inplace=True)
    print(df_lagged.head())
    return df_lagged


def train_test_final_pls(x_test,y_test,pls,X_mean, X_std,Y_mean, Y_std,Y_train_std_mean):
    x_col=x_test.columns
    X_test_std = (x_test.values - X_mean) / X_std
    Y_test_std = (y_test.values.reshape(-1, 1) - Y_mean) / Y_std
    Y_test_std = Y_test_std.ravel()
    y_pred_array=testing_timeseries_(X_test_std,pls,x_col,simple=True,number_lags=5)
    k1 = np.sum((Y_test_std - y_pred_array) ** 2)
    k2 = np.sum((Y_test_std - Y_train_std_mean) ** 2)
    q2 = float(1 - k1 / k2)
    #Y_pred_std = pls.predict(X_test_std).ravel()
    Y_pred = inverse_standardize(y_pred_array, Y_mean, Y_std)
    # Evaluate the model
    mse = mean_squared_error(y_test, Y_pred)
    r2 = r2_score(y_test, Y_pred)
    print(f"Q²(CV) value is {q2}")
    print(f"MSE value is {mse}")
    print(f"R² value is {r2}")
    """plt.figure(figsize=(10, 6))
    plt.scatter(y_test, Y_pred, alpha=0.7)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('PLS Regression: Actual vs Predicted')
    plt.grid()
    plt.show()"""
    return q2, mse, r2





def control_training_test_data():
    df1=pd.read_csv("df_block110_09_11_55_21.csv")# loading in the dataframes
    df2=pd.read_csv("df_block210_09_11_55_21.csv")
    df3=pd.read_csv("df_block310_09_11_55_21.csv")
    df4=pd.read_csv("df_block410_09_11_55_21.csv")
    # lagging the variables
    # Hyperparamter Laggingrule
    para_1=10
    lag_rule=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, para_1]#[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 9]
    df1_lagged=get_lagged_data(df1, lag_rule)#df1#
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:,:-1], df1_lagged.iloc[:,-1]
    df2_lagged=get_lagged_data(df2, lag_rule)#df2#
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:,:-1], df2_lagged.iloc[:,-1]
    df3_lagged=get_lagged_data(df3, lag_rule)#df3#
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:,:-1], df3_lagged.iloc[:,-1]
    df4_lagged=get_lagged_data(df4, lag_rule)#df4#
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:,:-1], df4_lagged.iloc[:,-1]
    # setting df4 and df1 as test
    # we are using df2 and df3 as trainining for the cross validation we assume that one is fully used and the parts of other one is used for validation and aft
    print(df2_lagged_x,df2_lagged_y)
    max_components_Lv_=15
    results=[]
    test_size=100
    for max_components_Lv in range(1,min(sum(lag_rule),max_components_Lv_)):
        X_train_1=pd.concat([df2_lagged_x, df4_lagged_x], ignore_index=True)
        Y_train_1 = pd.concat([df2_lagged_y, df4_lagged_y], ignore_index=True)
        pls_1, x_rs_1, y_res_1,mean_y_1=pls_regression_adj_cv(X_train_1, Y_train_1, max_components_Lv,initial_train_size=df2_lagged_x.shape[0]+test_size,test_size=test_size, step_size=test_size,number_lags=para_1,name_y="% Silica Concentrate_mean_lag")
        # changing 2 and 3 so 2 is now used for validation
        X_train_2 = pd.concat([df3_lagged_x, df2_lagged_x], ignore_index=True)
        Y_train_2 = pd.concat([df3_lagged_y, df2_lagged_y], ignore_index=True)
        pls_2, x_rs_2, y_res_2,mean_y_2=pls_regression_adj_cv(X_train_2, Y_train_2, max_components_Lv, initial_train_size=df4_lagged_x.shape[0] + test_size,
                           test_size=test_size, step_size=test_size, number_lags=para_1, name_y="% Silica Concentrate_mean_lag")
        # Percentage of variance explained in X and Y
        """X_var = np.var(pls_2.x_scores_, axis=0)
        X_total_var = np.var(X, axis=0).sum()
        pctvar_X = X_var / X_total_var * 100

        Y_var = np.var(pls.y_scores_, axis=0)
        Y_total_var = np.var(y, axis=0).sum()
        pctvar_Y = Y_var / Y_total_var * 100"""


        #plot_pls_coefficients_direct(pls_1,df1_lagged_x)
        #Testing
        # splitting x and y values
        #train_test_final_pls(df3_lagged_x,df3_lagged_y,pls_1,x_rs_1[0],x_rs_1[1],y_res_1[0],y_res_1[1],mean_y_1)
        results.append([max_components_Lv,train_test_final_pls(df1_lagged_x, df1_lagged_y, pls_2, x_rs_2[0], x_rs_2[1], y_res_2[0], y_res_2[1], mean_y_1)])
    print(results)
    n_components=[]
    q2_scores=[]
    mse_scores=[]
    r2_scores=[]
    for i in results:
        n_components.append(i[0])
        q2_scores.append(i[1][0])
        mse_scores.append(i[1][1])
        r2_scores.append(i[1][2])
    print(n_components, q2_scores)


    plt.figure(figsize=(8, 5))
    plt.plot(n_components, q2_scores, marker='o', linestyle='-', color='blue')
    plt.plot(n_components, mse_scores, marker='o', linestyle='-', color='green')
    plt.plot(n_components, r2_scores, marker='o', linestyle='-', color='red')

    plt.title("Q² vs Number of Latent Variables (LVs)")
    plt.xlabel("Number of Latent Variables")
    plt.ylabel("Q² Score")
    plt.xticks(np.asarray(n_components).astype(int))  # Ensure integer ticks
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    control_training_test_data()
"""#Use with raw data
df = pd.read_csv('MiningProcess_Flotation_Plant_Database.csv', decimal=',')
target_variable = '% Silica Feed'  # Replace with your target variable
pls_model, scaler_X, scaler_Y = pls_regression(df, target_variable, n_lags=3, n_components=20)
plot_pls_coefficients(pls_model, df, n_lags=3)
plot_q2_press_vs_components(df, target_variable, n_lags=3, max_components=20)
pls_model2, scaler_X2, scaler_Y2 = pls_regression(df, target_variable, n_lags=3, n_components=4)
plot_pls_coefficients(pls_model2, df, n_lags=3)
plot_q2_press_vs_components(df, target_variable, n_lags=3, max_components=4)"""


