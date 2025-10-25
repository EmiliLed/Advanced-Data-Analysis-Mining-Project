import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from datetime import datetime

from PLS_clean import plot_q2_press_vs_components

now = datetime.now()
import PLS_clean as PLS
import kpls


# --- Data Preparation ---

def get_lagged_data(df_original_norm, lagging_rule):  # lag data according to rules
    df_lagged = pd.DataFrame(index=df_original_norm.index)
    for index, key in enumerate(df_original_norm.columns):
        for i in range(lagging_rule[index]):
            lag = i + 1
            df_lagged[f'{key}_lag{lag}'] = df_original_norm[key].shift(lag)
    df_lagged[f"y_goal_{df_original_norm.keys()[-1]}"] = df_original_norm[df_original_norm.keys()[-1]]
    df_lagged.dropna(inplace=True)
    # print(df_lagged.head())
    return df_lagged


def load_data_blocks_and_split(rule_apply=True,
                               lag_rule=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6, 9]):
    df1 = pd.read_csv("df_block110_09_11_55_21.csv")  # loading in the dataframes, change to local name
    df2 = pd.read_csv("df_block210_09_11_55_21.csv")
    df3 = pd.read_csv("df_block310_09_11_55_21.csv")
    df4 = pd.read_csv("df_block410_09_11_55_21.csv")

    if rule_apply:
        df1_lagged = get_lagged_data(df1, lag_rule)
        df2_lagged = get_lagged_data(df2, lag_rule)
        df3_lagged = get_lagged_data(df3, lag_rule)
        df4_lagged = get_lagged_data(df4, lag_rule)
    else:
        df1_lagged = df1
        df2_lagged = df2
        df3_lagged = df3
        df4_lagged = df4

    return df1_lagged, df2_lagged, df3_lagged, df4_lagged


def testing_timeseries_(x, model, x_columns, number_lags=9, name_y="% Silica Concentrate_mean_lag", simple=True):
    y_pred_array = []
    y_pred_to_x_array = []
    if not simple:
        for index, x_values in enumerate(x):
            for i in range(len(y_pred_to_x_array)):
                x_values[x_columns.get_loc(name_y + str(i + 1))] = y_pred_to_x_array[i]
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


def model_testing(X_train, X_test, y_train, y_test, max_components_Lv, simple=True, model=None):
    x_col = X_test.columns
    X_train_std, X_mean, X_std = PLS.standardize(X_train.values)
    Y_train_std, Y_mean, Y_std = PLS.standardize(np.asarray(y_train).reshape(-1, 1))
    X_train_std = (X_train.values - X_mean) / X_std
    Y_train_std = (y_train.values.reshape(-1, 1) - Y_mean) / Y_std
    Y_train_std = Y_train_std.ravel()
    X_test_std = (X_test.values - X_mean) / X_std
    Y_test_std = (y_test.values.reshape(-1, 1) - Y_mean) / Y_std
    Y_test_std = Y_test_std.ravel()
    pls = PLSRegression(n_components=max_components_Lv)  # alternativly use direct model
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
    # plt.show()

    # Model evaluation
    mse = mean_squared_error(Y_test_std, y_pred_array)
    r2 = r2_score(Y_test_std, y_pred_array)
    print(f'Mean Squared Error: {mse}')
    print(f'R² Score: {r2}')
    return pls, (X_mean, X_std), (Y_mean, Y_std), mse, r2


def execute_modelling():
    # Hyperparamters
    para_1 = 50  # 9
    lag_rule = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 6,
                para_1]  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,para_1]
    max_components_Lv_ = 15
    df1_lagged, df2_lagged, df3_lagged, df4_lagged = load_data_blocks_and_split(lag_rule=lag_rule)
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]

    results = []
    variance_explained_x = []
    variance_explained_y = []
    test_size = 20

    for max_components_Lv in range(1, min(sum(lag_rule), max_components_Lv_)):
        X_train_1 = pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
        Y_train_1 = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
        pls_1, x_rs_1, y_res_1, mean_y_1, q2_array_1, mse_array_1, r2_array_1 = PLS.pls_regression_adj_cv(X_train_1,
                                                                                                          Y_train_1,
                                                                                                          max_components_Lv,
                                                                                                          initial_train_size=
                                                                                                          df2_lagged_x.shape[
                                                                                                              0] + test_size,
                                                                                                          test_size=test_size,
                                                                                                          step_size=test_size,
                                                                                                          number_lags=para_1,
                                                                                                          name_y="% Silica Concentrate_mean_lag")
        # changing 2 and 3 so 2 is now used for validation
        X_train_2 = pd.concat([df3_lagged_x, df2_lagged_x], ignore_index=True)
        Y_train_2 = pd.concat([df3_lagged_y, df2_lagged_y], ignore_index=True)
        pls_2, x_rs_2, y_res_2, mean_y_2, q2_array_2, mse_array_2, r2_array_2 = PLS.pls_regression_adj_cv(X_train_2,
                                                                                                          Y_train_2,
                                                                                                          max_components_Lv,
                                                                                                          initial_train_size=
                                                                                                          df2_lagged_x.shape[
                                                                                                              0] + test_size,
                                                                                                          test_size=test_size,
                                                                                                          step_size=test_size,
                                                                                                          number_lags=para_1,
                                                                                                          name_y="% Silica Concentrate_mean_lag")
        # Percentage of variance explained in X and Y simular to
        X_var = np.var(pls_2.x_scores_, axis=0)
        X_total_var = np.var(X_train_2, axis=0).sum()
        pctvar_X = X_var / X_total_var * 100
        variance_explained_x.append([max_components_Lv, pctvar_X])

        Y_var = np.var(pls_2.y_scores_, axis=0)
        Y_total_var = np.var(Y_train_2, axis=0).sum()
        pctvar_Y = Y_var / Y_total_var * 100

        variance_explained_y.append([max_components_Lv, pctvar_Y])
        q2_mean = np.mean(np.vstack((np.asarray(q2_array_1), np.asarray(q2_array_2)))[:, 3].mean())
        r2_mean = np.mean(np.vstack((np.asarray(r2_array_1), np.asarray(r2_array_2)))[:, 3].mean())
        mse_mean = np.mean(np.vstack((np.asarray(mse_array_1), np.asarray(mse_array_2)))[:, 3].mean())
        results.append([max_components_Lv, q2_mean, r2_mean, mse_mean])

        print(q2_array_1, q2_array_2)
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
    plt.plot(n_components, q2_scores, marker='o', linestyle='-', color='blue', label="Q2")
    plt.plot(n_components, mse_scores, marker='o', linestyle='-', color='green', label="MSE")
    plt.plot(n_components, r2_scores, marker='o', linestyle='-', color='red', label="R2")

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

def kpls_helper(X_train,Y_train,initial_train_size,test_size,step_size,model):
    q2_array = []
    r2_array = []
    mse_array = []
    if X_train.shape[0] < initial_train_size:
        raise ValueError('Initial training size is too small')
        return [],[],[]
    n_samples = X_train.shape[0]
    for start in range(initial_train_size, n_samples - test_size + 1, step_size):
        train_start = start - initial_train_size  # alternavily for extended window set 0
        train_end = start
        test_end = start + test_size
        X_train_help = X_train.values[train_start:train_end]
        y_train_help = Y_train.values[train_start:train_end]
        X_validation = X_train.values[train_end:test_end]
        y_validation = Y_train.values[train_end:test_end]
        model.fit(X_train_help, y_train_help)
        y_pred = model.predict(X_validation).ravel()
        k1 = np.sum((y_validation - y_pred) ** 2)
        k2 = np.sum((y_validation - np.mean(Y_train.values[:])) ** 2)
        q2 = 1 - k1 / k2
        q2_array.append([train_start, train_end, test_end, q2])

        # Evaluate the model
        mse = mean_squared_error(y_validation , y_pred)
        r2 = r2_score(y_validation , y_pred)
        mse_array.append([train_start, train_end, test_end, mse])
        r2_array.append([train_start, train_end, test_end, r2])
    return q2_array, r2_array, mse_array

def execute_modelling_kpls():
    # Hyperparamters
    para_1 = 50  # 9
    lag_rule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                2]  # [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6,para_1]
    max_components_Lv_ = 3
    df1_lagged, df2_lagged, df3_lagged, df4_lagged = load_data_blocks_and_split(lag_rule=lag_rule)
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]

    results = []
    test_size = 20
    initial_train_size = 400
    step_size = 20

    for max_components_Lv in range(1, min(sum(lag_rule), max_components_Lv_)):
        X_train_1 = pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
        Y_train_1 = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
        X_train_2 = pd.concat([df3_lagged_x, df2_lagged_x], ignore_index=True)
        Y_train_2 = pd.concat([df3_lagged_y, df2_lagged_y], ignore_index=True)
        model = kpls.KPLS(n_components=max_components_Lv, kernel="linear")
        q2_array_1, r2_array_1, mse_array_1 = kpls_helper(X_train=X_train_1, Y_train=Y_train_1, initial_train_size=initial_train_size,test_size=test_size,step_size=step_size,model=model)
        q2_array_2, r2_array_2, mse_array_2=kpls_helper(X_train=X_train_2, Y_train=Y_train_2, initial_train_size=initial_train_size,test_size=test_size,step_size=step_size,model=model)

        q2_mean = np.mean(np.vstack((np.asarray(q2_array_1), np.asarray(q2_array_2)))[:, 3].mean())
        r2_mean = np.mean(np.vstack((np.asarray(r2_array_1), np.asarray(r2_array_2)))[:, 3].mean())
        mse_mean = np.mean(np.vstack((np.asarray(mse_array_1), np.asarray(mse_array_2)))[:, 3].mean())
        results.append([max_components_Lv, q2_mean, r2_mean, mse_mean])

        print(q2_array_1, q2_array_2)
        print(f"Q² mean{q2_mean}")
        #print(f"r2 mean{r2_mean}")
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
    plt.plot(n_components, q2_scores, marker='o', linestyle='-', color='blue', label="Q2")
    plt.plot(n_components, mse_scores, marker='o', linestyle='-', color='green', label="MSE")
    plt.plot(n_components, r2_scores, marker='o', linestyle='-', color='red', label="R2")

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
    max_components_Lv = 6;
    start_rule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 5]
    df1_lagged, df2_lagged, df3_lagged, df4_lagged = load_data_blocks_and_split(lag_rule=start_rule)
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]
    X_train = pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
    Y_train = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
    X_test_1 = pd.concat([df4_lagged_x, df1_lagged_x], ignore_index=True)
    Y_test_1 = pd.concat([df4_lagged_y, df1_lagged_y], ignore_index=True)
    # model_testing(X_train,df4_lagged_x,Y_train,df4_lagged_y,max_components_Lv,simple=False)
    # model_testing(X_train, df1_lagged_x, Y_train, df1_lagged_y, max_components_Lv, simple=False)
    max_components_Lv = min(max_components_Lv, sum(start_rule))
    pls, (X_mean, X_std), (Y_mean, Y_std), mse, r2 = model_testing(X_train, X_test_1, Y_train, Y_test_1,
                                                                   max_components_Lv)
    print(r2)


def execute_testing_kpls():
    max_components_Lv = 6;
    start_rule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
    df1_lagged, df2_lagged, df3_lagged, df4_lagged = load_data_blocks_and_split(lag_rule=start_rule)
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]

    X_train = pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
    Y_train = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
    X_test_1 = pd.concat([df4_lagged_x, df1_lagged_x], ignore_index=True)
    Y_test_1 = pd.concat([df4_lagged_y, df1_lagged_y], ignore_index=True)

    # model_testing(X_train,df4_lagged_x,Y_train,df4_lagged_y,max_components_Lv,simple=False)
    # model_testing(X_train, df1_lagged_x, Y_train, df1_lagged_y, max_components_Lv, simple=False)
    max_components_Lv = min(max_components_Lv, sum(start_rule))
    print("in")
    model = kpls.KPLS(n_components=max_components_Lv, kernel='rbf', kernel_params={"gamma": 0.01})
    model.fit(X_train, Y_train)
    print("back")
    y_pred = model.predict(X_test_1)
    print("back")
    mse = mean_squared_error(Y_test_1, y_pred)
    r2 = r2_score(Y_test_1, y_pred)
    print(r2)
    print(mse)


def execute_testing_kpls_diff_kernels():
    max_components_Lv = 6
    start_rule = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2]

    df1_lagged, df2_lagged, df3_lagged, df4_lagged = load_data_blocks_and_split(lag_rule=start_rule)
    df1_lagged_x, df1_lagged_y = df1_lagged.iloc[:, :-1], df1_lagged.iloc[:, -1]
    df2_lagged_x, df2_lagged_y = df2_lagged.iloc[:, :-1], df2_lagged.iloc[:, -1]
    df3_lagged_x, df3_lagged_y = df3_lagged.iloc[:, :-1], df3_lagged.iloc[:, -1]
    df4_lagged_x, df4_lagged_y = df4_lagged.iloc[:, :-1], df4_lagged.iloc[:, -1]

    X_train = pd.concat([df2_lagged_x, df3_lagged_x], ignore_index=True)
    Y_train = pd.concat([df2_lagged_y, df3_lagged_y], ignore_index=True)
    X_test_1 = pd.concat([df4_lagged_x, df1_lagged_x], ignore_index=True)
    Y_test_1 = pd.concat([df4_lagged_y, df1_lagged_y], ignore_index=True)

    max_components_Lv = min(max_components_Lv, sum(start_rule))

    # test different kernels
    kernels = {
        "rbf": [
            {"gamma": 0.01},
            {"gamma": 0.1},
            {"gamma": 0.5}
        ],
        "poly": [
            {"degree": 2, "coef0": 0, "gamma": 0.01},
            {"degree": 3, "coef0": 1, "gamma": 0.01}
        ],
        "sigmoid": [
            {"gamma": 0.01, "coef0": 0.1},
            {"gamma": 0.05, "coef0": 0.5},
        ],
        "linear": [
            {}  # No parameters for linear kernel
        ]
    }

    q2_scores = []

    for kernel_name, param_list in kernels.items():
        for params in param_list:
            print(f"Fitting KPLS with kernel = {kernel_name} with params = {params}")
            model = kpls.KPLS(n_components=max_components_Lv, kernel=kernel_name, kernel_params=params)
            model.fit(X_train, Y_train)
            y_pred = model.predict(X_test_1)

            # compute mse r2 q2
            mse = mean_squared_error(Y_test_1, y_pred)
            r2 = r2_score(Y_test_1, y_pred)
            ss_res = np.sum((Y_test_1 - y_pred.flatten()) ** 2)
            ss_tot = np.sum((Y_test_1 - np.mean(Y_test_1)) ** 2)
            q2 = 1 - (ss_res / ss_tot)

            print(f"Kernel: {kernel_name} | R²: {r2:.4f} | MSE: {mse:.4f} | Q²: {q2:.4f}")
            q2_scores.append(q2)

    # plot Q2 accross kernels
    # Kernel and grouping setup
    kernel_names = list(kernels.keys())
    group_sizes = [len(v) for v in kernels.values()]
    max_group_size = max(group_sizes)
    indices = np.arange(len(kernel_names))
    bar_width = 0.15

    # Prepare bar heights
    plot_data = np.zeros((max_group_size, len(kernel_names)))
    score_idx = 0
    for col, size in enumerate(group_sizes):
        for row in range(size):
            plot_data[row, col] = q2_scores[score_idx]
            score_idx += 1

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = []
    for i in range(max_group_size):
        b = ax.bar(indices + i * bar_width, plot_data[i], width=bar_width, label=f'Set {i + 1}')
        bars.append(b)

    ax.set_xlabel("Kernel Type")
    ax.set_ylabel("Q² (Predictive Ability)")
    ax.set_title("K-PLS Q² Comparison Across Kernels and Parameter Sets")
    ax.set_xticks(indices + bar_width * (max_group_size - 1) / 2)
    ax.set_xticklabels(kernel_names)
    ax.set_ylim(0, 1)
    ax.legend(title="Parameter Sets")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add parameter text labels to bars
    score_idx = 0
    for col, (kernel, param_list) in enumerate(kernels.items()):
        for row, params in enumerate(param_list):
            label_text = ', '.join([f'{k}={v}' for k, v in params.items()]) if params else "default"
            height = plot_data[row, col]
            if height > 0:
                ax.text(
                    x=col + row * bar_width - 0.1,  # position near bar center
                    y=height + 0.02,
                    s=label_text,
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    rotation=45
                )
            score_idx += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # execute_modelling()
    #execute_testing_kpls_diff_kernels()
    execute_modelling_kpls()
