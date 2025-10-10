import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
now = datetime.now()
import PLS_clean as PLS


# --- Data Preparation ---

def get_lagged_data(df_original_norm,lagging_rule): # lag data according to rules
    df_lagged = pd.DataFrame(index=df_original_norm.index)
    for index,key in enumerate(df_original_norm.columns):
        for i in range(lagging_rule[index]):
            lag=i+1
            df_lagged[f'{key}_lag{lag}'] = df_original_norm[key].shift(lag)
    df_lagged[f"y_goal_{df_original_norm.keys()[-1]}"] = df_original_norm[df_original_norm.keys()[-1]]
    df_lagged.dropna(inplace=True)
    print(df_lagged.head())
    return df_lagged


def load_data_blocks_and_split(rule_apply=True,lag_rule=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 9]):
    df1 = pd.read_csv("df_block110_09_11_55_21.csv")  # loading in the dataframes, change to local name
    df2 = pd.read_csv("df_block210_09_11_55_21.csv")
    df3 = pd.read_csv("df_block310_09_11_55_21.csv")
    df4 = pd.read_csv("df_block410_09_11_55_21.csv")

    if rule_apply:
        df1_lagged=get_lagged_data(df1, lag_rule)
        df2_lagged = get_lagged_data(df2, lag_rule)
        df3_lagged = get_lagged_data(df3, lag_rule)
        df4_lagged=get_lagged_data(df4, lag_rule)
    else:
        df1_lagged=df1
        df2_lagged=df2
        df3_lagged=df3
        df4_lagged=df4

    return df1_lagged, df2_lagged, df3_lagged, df4_lagged


def testing_timeseries_(x,model,x_columns,number_lags=9,name_y="% Silica Concentrate_mean_lag",simple=True):
    y_pred_array = []
    y_pred_to_x_array = []
    if not simple:
        for index, x_values in enumerate(x):
            for i in range(len(y_pred_to_x_array)):
                x_values[x_columns.get_loc(name_y + str(i+1))] = y_pred_to_x_array[i]
            y_pred = model.predict(x_values.reshape(1, -1)).item()
            y_pred_array.append(y_pred)
            y_pred_to_x_array = [y_pred] + y_pred_to_x_array
            while len(y_pred_to_x_array) > number_lags:
                y_pred_to_x_array = y_pred_to_x_array[:number_lags]

        return y_pred_array
    else:
        for index, x_values in enumerate(x):
            y_pred = model.predict(x_values.reshape(1, -1)).item()
            y_pred_array.append(y_pred)
        return y_pred_array


def model_testing(X_train,X_test,y_train,y_test,max_components_Lv,simple=True,model=None):
    x_col = X_test.columns
    X_train_std, X_mean, X_std = PLS.standardize(X_train.values)
    Y_train_std, Y_mean, Y_std = PLS.standardize(np.asarray(y_train).reshape(-1, 1))
    X_train_std = (X_train.values - X_mean) / X_std
    Y_train_std = (y_train.values.reshape(-1, 1) - Y_mean) / Y_std
    Y_train_std = Y_train_std.ravel()
    X_test_std = (X_test.values - X_mean) / X_std
    Y_test_std = (y_test.values.reshape(-1, 1) - Y_mean) / Y_std
    Y_test_std = Y_test_std.ravel()
    pls = PLSRegression(n_components=max_components_Lv)# alternativly use direct model
    pls.fit(X_train_std, Y_train_std)
    if model is not None:
        y_pred_array = testing_timeseries_(X_test_std, model, x_col, simple=True, number_lags=9)
    else:
        y_pred_array = testing_timeseries_(X_test_std, pls, x_col, simple=simple, number_lags=9)

    plt.figure(figsize=(10, 6))
    plt.scatter(Y_test_std, y_pred_array, alpha=0.7)
    plt.plot([Y_test_std.min(), Y_test_std.max()], [Y_test_std.min(), Y_test_std.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('PLS Regression: Actual vs Predicted')
    plt.grid()
    plt.show()

    #Model evaluation
    mse = mean_squared_error(Y_test_std, y_pred_array)
    r2 = r2_score(Y_test_std, y_pred_array)
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    return pls,(X_mean,X_std),(Y_mean,Y_std)


def execute_modelling():
    # Hyperparamters
    para_1 = 50#9
    lag_rule =  [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, para_1]#[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,para_1]
    max_components_Lv_ = 15
    df1_lagged, df2_lagged, df3_lagged, df4_lagged=load_data_blocks_and_split(lag_rule=lag_rule)
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]

    results = []
    variance_explained_x=[]
    variance_explained_y=[]
    test_size = 20

    for max_components_Lv in range(1,min(sum(lag_rule),max_components_Lv_)):

        X_train_1=pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
        Y_train_1 = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
        pls_1, x_rs_1, y_res_1,mean_y_1,q2_array_1,mse_array_1,r2_array_1=PLS.pls_regression_adj_cv(X_train_1, Y_train_1, max_components_Lv,initial_train_size=df2_lagged_x.shape[0]+test_size,test_size=test_size, step_size=test_size,number_lags=para_1,name_y="% Silica Concentrate_mean_lag")
        # changing 2 and 3 so 2 is now used for validation
        X_train_2 = pd.concat([df3_lagged_x, df2_lagged_x], ignore_index=True)
        Y_train_2 = pd.concat([df3_lagged_y, df2_lagged_y], ignore_index=True)
        pls_2, x_rs_2, y_res_2,mean_y_2,q2_array_2,mse_array_2,r2_array_2=PLS.pls_regression_adj_cv(X_train_2, Y_train_2, max_components_Lv, initial_train_size=df2_lagged_x.shape[0] + test_size,
                           test_size=test_size, step_size=test_size, number_lags=para_1, name_y="% Silica Concentrate_mean_lag")
        # Percentage of variance explained in X and Y simular to
        X_var = np.var(pls_2.x_scores_, axis=0)
        X_total_var = np.var(X_train_2, axis=0).sum()
        pctvar_X = X_var / X_total_var * 100
        variance_explained_x.append([max_components_Lv,pctvar_X])

        Y_var = np.var(pls_2.y_scores_, axis=0)
        Y_total_var = np.var(Y_train_2, axis=0).sum()
        pctvar_Y = Y_var / Y_total_var * 100

        variance_explained_y.append([max_components_Lv,pctvar_Y])
        q2_mean=np.mean(np.vstack((np.asarray(q2_array_1),np.asarray(q2_array_2)))[:,3].mean())
        r2_mean = np.mean(np.vstack((np.asarray(r2_array_1), np.asarray(r2_array_2)))[:, 3].mean())
        mse_mean = np.mean(np.vstack((np.asarray(mse_array_1), np.asarray(mse_array_2)))[:, 3].mean())
        results.append([max_components_Lv,q2_mean,r2_mean,mse_mean])
        
        print(q2_array_1,q2_array_2)
        print(f"Q² mean{q2_mean}")
        print(f"r2 mean{r2_mean}")
        print(f"mse mean{mse_mean}")

    print(results)
    n_components = []
    q2_scores = []
    mse_scores = []
    r2_scores = []
    for i in results:
        n_components.append(i[0])
        q2_scores.append(i[1])
        mse_scores.append(i[3])
        r2_scores.append(i[2])
    plt.figure(figsize=(8, 5))
    plt.plot(n_components, q2_scores, marker='o', linestyle='-', color='blue',label="Q2")
    plt.plot(n_components, mse_scores, marker='o', linestyle='-', color='green',label="MSE")
    plt.plot(n_components, r2_scores, marker='o', linestyle='-', color='red',label="R2")

    """print(results)
    
    for i in results:
        n_components.append(i[0])
        q2_scores.append(i[1][0])
        mse_scores.append(i[1][1])
        r2_scores.append(i[1][2])
    print(n_components, q2_scores)"""


    plt.title("Q² vs Number of Latent Variables (LVs)")
    plt.xlabel("Number of Latent Variables")
    plt.ylabel("Q² Score")
    plt.xticks(np.asarray(n_components).astype(int))  # Ensure integer ticks
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def execute_testing():
    max_components_Lv=6;
    df1_lagged, df2_lagged, df3_lagged, df4_lagged = load_data_blocks_and_split()
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]
    X_train = pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
    Y_train = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
    X_test_1 = pd.concat([df4_lagged_x, df1_lagged_x], ignore_index=True)
    Y_test_1 = pd.concat([df4_lagged_y, df1_lagged_y], ignore_index=True)
    #model_testing(X_train,df4_lagged_x,Y_train,df4_lagged_y,max_components_Lv,simple=False)
    #model_testing(X_train, df1_lagged_x, Y_train, df1_lagged_y, max_components_Lv, simple=False)
    model_testing(X_train,X_test_1,Y_train,Y_test_1,max_components_Lv)

if __name__=='__main__':
    #execute_modelling()
    execute_testing()
