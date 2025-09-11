import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import scipy as sp
now = datetime.now()
print("Current timestamp:", now)


def load_initial_data():
    return pd.read_csv("MiningProcess_Flotation_Plant_Database.csv")

def look_for_data_impurities(df):
    return
def look_for_outliers(df,keys):
    for i in keys:
        lower_bbound_q1 = df[i].quantile(0.25)
        higher_bbound_q1 = df[i].quantile(0.75)
        IQR = higher_bbound_q1 - lower_bbound_q1
        lower_bound = lower_bbound_q1 - 1.5 * IQR
        upper_bound = higher_bbound_q1 + 1.5 * IQR
        outliers = df[(df[i] < lower_bound) | (df[i] > upper_bound)]

        print(f"Outliers in '{i}':")
        print(outliers)
    return
def data_cleaning(df):
    return
def detrmin_data_distribuntion():
    return
def convert_to_time_series():
    return

def plot_df(df):
    return
def create_df_dates(df,save=False):
    df_1 = df['date'].value_counts()
    max_val=df['date'].max()
    min_val=df['date'].min()
    print(df_1)
    print(max_val,min_val)
    timestamps = pd.date_range(start=min_val, end=max_val, freq='H').astype(str)
    x = []
    y = []
    for i in timestamps:
        print(i, type(i))
        val = df_1.get(i, default=0)
        print(i, " ", val)
        x.append(i)
        y.append(val)

    df_dates = pd.DataFrame({'dates': x, 'counts': y})
    df_dates.to_csv("mining_process_dates_sorted" + '.csv', index=False)
    if save:
        df_dates.to_csv("mining_process_dates_counts" + now.strftime("%m_%d_%H_%M_%S") + '.csv', index=False)
    return df_dates
def create_mean_std(df,df_dates,save=False):
    df_help = df.copy()
    keys_normal = df.keys()[1:]
    new_keys = [f'{i}_mean' for i in keys_normal]
    new_keys_ = [f'{i}_std' for i in keys_normal]
    new_keys = np.concatenate((new_keys, new_keys_))
    print(new_keys)
    #######
    df_new = df_dates
    for col in new_keys:
        df_new[col] = np.nan
    cols_to_update = df_new.columns[2:]
    for i in df_new['dates']:
        if df_new[df_new['dates'] == i]["counts"].values[0] == 0:
            continue
        matching_rows = df[df['date'] == i]
        # print(matching_rows.iloc[:, 1:])
        matching_rows_ = matching_rows.iloc[:, 1:]
        means = matching_rows_.mean()
        stds = matching_rows_.std()
        means.index = [f"{col}_mean" for col in means.index]
        stds.index = [f"{col}_std" for col in stds.index]
        summary_row = pd.concat([means, stds])
        list_keys = summary_row.index.to_list
        index = df_new[df_new['dates'] == i].index
        # help_value=summary_row.index()
        for k in new_keys:
            df_new.loc[index, k] = summary_row.loc[k]
    if save:
        df_help.to_csv("mining_process_means"+now.strftime("%m_%d_%H_%M_%S")+'.csv', index=False)
    return df_help
def func_visulized_df(df,keys):# df = pd.read_csv("mining_process_means09_09_10_07_21.csv")keys=df.keys()[1:25]
    for i in range(0, len(keys)):
        print(keys[i])
        plt.plot(df.index, df[keys[i]], marker='.', linestyle='-')
        plt.xlabel("index")
        plt.ylabel(keys[i])
        plt.title('Distribution of Value')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def data_visuliying_each_column(df_in,keys,distribution=True, boxplot=False,correlation=False,spectral=False ):
    df=df_in.copy()
    if distribution:
        for col in keys:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of Numeric Column: {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
    if boxplot:
        for col in keys:
            plt.figure(figsize=(6, 4))
            sns.boxplot(x=df[col])
            plt.title(f'Boxplot of {col}')
            plt.xlabel(col)
            plt.grid(True)
            plt.show()
    if correlation:
        print(df)
        print(df.dtypes)
        co_mtx = df.corr(numeric_only=True)
        # Print correlation matrix
        print(co_mtx)
        #sns.heatmap(co_mtx, cmap="YlGnBu", annot=True)
        #plt.show()
    if spectral:
        plt.figure(figsize=(10, 6))
        for index, row in df.iterrows():
            plt.plot(df.columns, row, label=f'Row {index}')
        plt.xlabel('Columns (Features)')
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.legend(loc='best', bbox_to_anchor=(1.05, 1), fontsize='small')
        plt.show()

def expiriment_test(df):
    df['Sum_Last_Two'] = df.iloc[:, -2] +df.iloc[:, -1]

    # Plot the result
    df['Sum_Last_Two'].plot(kind='line', marker='o', title='Sum of Last Two Columns')
    plt.xlabel('Index')
    plt.ylabel('Sum')
    plt.grid(True)
    plt.show()
def detailled_insight(df,key):
    smallest_100 = df[key].nsmallest(100)
    for idx in smallest_100.index:
        pos = df.index.get_loc(idx)
        prev_val = None
        next_val = None

        # Previous row value (if exists)
        if pos > 0:
            prev_idx = df.index[pos - 1]
            prev_val = df.loc[prev_idx, key]

        # Next row value (if exists)
        if pos < len(df) - 1:
            next_idx = df.index[pos + 1]
            next_val = df.loc[next_idx, key]

        # Append result as tuple: (current_value, prev_value, next_value)
        print((idx,df["date"].loc[idx] ,df[key].loc[idx], prev_val, next_val))
def find_redunant_ranges(df,key):
    col = key # replace with your column name
    n = 8000  # number of subsequent values to check

    values = df[col].values
    length = len(values)

    indices = []
    i=0

    while i < length - n:
        # Check if all next n values are equal to current
        # Using numpy slicing and np.all for speed
        if (values[i] == values[i + 1:i + n + 1]).all():
            t=0
            while values[i] ==values[i + n + t]:
                t+=1
            count=n+t
            indices.append((df.index[i],count))
            i+=count
            continue
        i += 1
    print(f"Indices where next {n} values are the same as current value:")
    print(indices)


def plottin_floating_Average(df,key,n=1000):
    df_1=df.copy()
    print(df_1[key])

    df_2= df_1[key].rolling(window=n, center=True).mean()
    print(df_2)
    return
    plt.figure(figsize=(10, 5))
    plt.plot(df[key], label='Original', marker='o')
    plt.plot(df_2['moving_avg'], label=f'{n}-Point Moving Avg', linestyle='--', color='orange')

    plt.title(f'Moving Average (Window = {n})')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__=="__main__":
    df_= load_initial_data()
    df_.iloc[:, 1:] = df_.iloc[:, 1:].replace(',', '.', regex=True).astype(float)
    nan_counts = df_.isna().sum()
    df=df_.copy()
    for key in df.keys()[1:]:
        df[key] = pd.to_numeric(df[key], errors='coerce')

    print(df_.keys()[1:])
    print(df.iloc[:2, 1:5])
    print(df["Starch Flow"].nsmallest(100))
    detailled_insight(df, "Starch Flow")
    matching_rows = df[df['date'] == "2017-06-27 08:00:00"]
    pd.set_option('display.max_rows', None)
    print(matching_rows["Starch Flow"])
    pd.set_option('display.max_rows', 10)
    #data_visuliying_each_column(df.iloc[:, 1:],df.keys()[1:],distribution=False, boxplot=False, correlation=True, spectral=False)
    #func_visulized_df(df.iloc[:, 1:],df.keys()[1:])
    find_redunant_ranges(df, "% Iron Feed")
    plottin_floating_Average(df,"Flotation Column 05 Air Flow")
    #look_for_data_impurities(df_)
    #plot_df(df_)
