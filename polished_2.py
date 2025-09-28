import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import scipy as sp
from scipy.stats import f
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import polished_up_data_analysing as pol_1
now = datetime.now()
columns_keys = [
    'date',
    '% Iron Feed',
    '% Silica Feed',
    'Starch Flow',
    'Amina Flow',
    'Ore Pulp Flow',
    'Ore Pulp pH',
    'Ore Pulp Density',
    'Flotation Column 01 Air Flow',
    'Flotation Column 02 Air Flow',
    'Flotation Column 03 Air Flow',
    'Flotation Column 04 Air Flow',
    'Flotation Column 05 Air Flow',
    'Flotation Column 06 Air Flow',
    'Flotation Column 07 Air Flow',
    'Flotation Column 01 Level',
    'Flotation Column 02 Level',
    'Flotation Column 03 Level',
    'Flotation Column 04 Level',
    'Flotation Column 05 Level',
    'Flotation Column 06 Level',
    'Flotation Column 07 Level',
    '% Iron Concentrate',
    '% Silica Concentrate'
]
def find_redunant_ranges_and_set_nan(df,key):
    col = key # replace with your column name
    n = 20  # number of subsequent values to check

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
    for index in indices:
        start_pos = df.index.get_loc(index[0])

        # Get the index labels from position start_pos to start_pos + k
        affected_indices = df.index[start_pos: start_pos + index[1] + 1]

        # Replace values with NaN in the specified column
        df.loc[affected_indices, key] = np.nan
    return indices

import numpy as np
from scipy.stats import linregress

def detect_trend_segments(data, min_len=20, r_squared_thresh=0.99):
    segments = []
    n = len(data)
    for start in range(n - min_len + 1):
        for end in range(start + min_len, n + 1):
            y = data[start:end]
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            if abs(r_value) >= r_squared_thresh:
                direction = "rising" if slope > 0 else "falling"
                segments.append({
                    "start": start,
                    "end": end,
                    "slope": slope,
                    "r_squared": r_value**2,
                    "direction": direction
                })
    print(segments)
    return segments
def lagged_data_correlation_multi_column(df):
    plotting=False
    columns = df.columns

    max_lag = 350
    lags = range(1, max_lag + 1)

    special_key=df.keys()[-1]
    col2=special_key
    col_lag_ret=[]
    for col1 in df.columns:
        max_lag=0
        max_lag_value = 0
        corr_values = []
        for lag in lags:
            df_1=df.copy()
            shifted_col = df_1[col1].shift(lag)
            new_df = pd.DataFrame({
                'col1_shifted': shifted_col,
                col2: df_1[col2]
            })
            new_df = new_df.dropna(subset=['col1_shifted'])
            corr = new_df['col1_shifted'].corr(new_df[col2])
            if abs(corr)>abs(max_lag_value):
                max_lag_value=corr
                max_lag=lag
                count=len(new_df)
            corr_values.append(corr)
        print(f"max lagged for {col1} is a lag{max_lag} with vale of {max_lag_value} with {count} rows")
        col_lag_ret.append([col1, max_lag])
        if not plotting:
            continue
        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(lags, corr_values, label=f'{col1} (lagged) vs {col2}', color='slateblue')
        plt.axhline(0, linestyle='--', color='gray')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.title(f'Lagged Cross-Correlation: {col1} (t-lag) → {col2} (t)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return col_lag_ret

def addaped_frequence_dates(df,save=False):
    print(df)
    max_val = df['dates'].max()
    min_val = df['dates'].min()
    print(max_val, min_val)
    timestamps = pd.date_range(start=min_val, end=max_val).astype(str)
    time_spanns = [["01:00:00", "02:00:00", "03:00:00", "04:00:00", "05:00:00", "06:00:00"],
                   ["07:00:00", "08:00:00", "09:00:00", "10:00:00", "11:00:00", "12:00:00", "13:00:00", "14:00:00",
                    "15:00:00"],
                   ["16:00:00", "17:00:00", "18:00:00", "19:00:00", "20:00:00", "21:00:00", "22:00:00", "23:00:00",
                    "00:00:00"]]

    df_low_freq= pd.DataFrame(columns=df.columns)
    for index,day in enumerate(timestamps):
        for index_block,hour_block in enumerate(time_spanns):
            df_row = pd.DataFrame([{}], columns=df.columns)
            df_row['dates'] = day+" "+hour_block[0]
            targets=[]
            for hours in hour_block:
                if hours=="00:00:00":
                    if index<(len(timestamps)-1):
                        key = timestamps[index+1] + " " + hours
                        targets.append(key)
                else:
                    key=day+" "+hours
                    targets.append(key)
            df_helper=df[df['dates'].astype(str).isin(targets)]
            print(df_helper)
            df_clean = df_helper.dropna(how='all')

            # Step 2: If the DataFrame is empty, add one row of NaNs
            if df_clean.empty:
                df_row = pd.DataFrame([[np.nan] * len(df.columns)], columns=df.columns)
                df_row['counts'] = 0
            else:
                total_n = df_clean['counts'].sum()
                df_row['counts'] = total_n
                for col in columns_keys[1:]:
                    mean_key=f"{col}_mean"
                    std_key=f"{col}_std"

                    combined_mean = (df_clean[mean_key] * df_clean['counts']).sum() / total_n
                    sum_of_squares = (
                            df_clean['counts'] * (df_clean[std_key] ** 2 + (df_clean[mean_key] - combined_mean) ** 2)
                    ).sum()
                    combined_std = np.sqrt(sum_of_squares / total_n)

                    # Result as a one-row DataFrame
                    df_row[mean_key]=combined_mean
                    df_row[std_key] = combined_std
            print(df_row["% Silica Concentrate_mean"])
            df_low_freq=pd.concat([df_low_freq, df_row], ignore_index=True)
    print(df_low_freq)
    print(df_low_freq.shape)
    df_dates=df_low_freq.copy()
    for key in df_low_freq.keys()[2:25]:
        find_redunant_ranges_and_set_nan(df_low_freq,key)

    lagged_data_correlation_multi_column(df_low_freq.iloc[:,2:25])
    #pol_1.func_visulized_df(df_low_freq,df_low_freq.keys()[2:3])
    if save:
        df_dates.to_csv("mining_process_dates_low_res" + now.strftime("%m_%d_%H_%M_%S") + '.csv', index=False)
    return

def centering_normal(df,keys):
    df_=df.copy()
    print(df_.keys())
    print(df_[keys[0]])
    for key in keys:
        df_help=df_[key].dropna(how='all')
        mean_=df_help.mean()
        std_=df_help.std()
        df_[key]=(df_help-mean_)/std_
    return df_


def addaped_frequence_dates_data_originale(save=False):
    df=pd.read_csv("MiningProcess_Flotation_Plant_Database.csv")
    df.iloc[:, 1:] = df.iloc[:, 1:].replace(',', '.', regex=True).astype(float)
    print(df)
    max_val = df['date'].max().split(' ')[0]
    min_val = df['date'].min().split(' ')[0]
    print(max_val, min_val)
    timestamps = pd.date_range(start=min_val, end=max_val).astype(str)
    time_spanns = [["01:00:00", "02:00:00", "03:00:00", "04:00:00", "05:00:00", "06:00:00"],
                   ["07:00:00", "08:00:00", "09:00:00", "10:00:00", "11:00:00", "12:00:00", "13:00:00", "14:00:00",
                    "15:00:00"],
                   ["16:00:00", "17:00:00", "18:00:00", "19:00:00", "20:00:00", "21:00:00", "22:00:00", "23:00:00",
                    "00:00:00"]]
    #new columns_names
    keys_new=["date","counts"]
    for col in df.keys()[1:]:
        keys_new.append(f"{col}_mean")
    print(keys_new)
    df_low_freq= pd.DataFrame(columns=keys_new)
    for index,day in enumerate(timestamps):
        for index_block,hour_block in enumerate(time_spanns):
            df_row = pd.DataFrame([{}], columns=keys_new)
            df_row['date'] = day+" "+hour_block[0]
            targets=[]
            for hours in hour_block:
                if hours=="00:00:00":
                    if index<(len(timestamps)-1):
                        key = timestamps[index+1] + " " + hours
                        targets.append(key)
                else:
                    key=day+" "+hours
                    targets.append(key)
            print(targets)
            df_helper=df[df['date'].astype(str).isin(targets)]
            print(df_helper)
            df_clean = df_helper.dropna(how='all')


            #Step 2: If the DataFrame is empty, add one row of NaNs
            if df_clean.empty:
                df_row = pd.DataFrame([[np.nan] * len(df_row.columns)], columns=df_row.columns)
                df_row['counts'] = 0
                df_row['date'] = day + " " + hour_block[0]
            else:

                df_row['counts'] = len(df_clean)
                for col in columns_keys[1:]:
                    mean_key=f"{col}_mean"
                    combined_mean = df_clean[col].mean()
                    df_row[mean_key]=combined_mean

            print(df_row["% Silica Concentrate_mean"])
            df_low_freq=pd.concat([df_low_freq, df_row], ignore_index=True)

    print(df_low_freq)
    print(df_low_freq.shape)
    #df_dates=centering_normal(df_low_freq,keys_new[2:])
    print(df_low_freq.keys()[2:25])
    for col in columns_keys[1:]:
        mean_key=f"{col}_mean"
        find_redunant_ranges_and_set_nan(df_low_freq,mean_key)

    lagged_data_correlation_multi_column(df_low_freq.iloc[:,2:25])
    #pol_1.func_visulized_df(df_low_freq,df_low_freq.keys()[2:3])
    if save:
        df_low_freq.to_csv("mining_process_dates_low_res_full" + now.strftime("%m_%d_%H_%M_%S") + '.csv', index=False)
    return
def set_indeces_of_col_to_nan(df,key,index_ranges):
    for start, end in index_ranges:
        df.loc[start:end+1, key] = np.nan
    return df
def lag_data_set(df,lag_info):
    shifted_data = pd.DataFrame(index=df.index)  # Keep full index

    for col_name, lag in lag_info:
        print(lag)
        new_col_name = f"{col_name}_lag{lag}"
        shifted_data[new_col_name] = df[col_name].shift(lag)

    return shifted_data
def spectral_print(df,title="Spectral Plot", cmap='viridis'):
    plt.figure(figsize=(12, 6))

    # Optional: color each line differently
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(df)))

    for i, row in enumerate(df.values):
        plt.plot(df.columns, row, color=colors[i], alpha=0.8)

    plt.title(title)
    plt.xlabel("Variables")
    plt.ylabel("Value")
    plt.grid(True)
    plt.xticks(rotation=90, ha='right')  # vertical and right-aligned
    plt.tight_layout()
    plt.show()

def data_preteating_1(df):
    print(df)
    keys_df=df.keys()[2:25]
    df_centered = centering_normal(df, keys_df)
    for key in df_centered.keys()[2:]:
        print(key)
        find_redunant_ranges_and_set_nan(df_centered, key)
    col_name="Flotation Column 04 Air Flow_mean"
    index_ranges = [
        (0, 135),
        (405, 413),
        (431, 467),
        (473, 479),
        (524, 535)
    ]
    set_indeces_of_col_to_nan(df_centered,col_name,index_ranges)
    col_name = "Flotation Column 05 Air Flow_mean"
    index_ranges = [
        (0, 135),
        (404, 413),
        (434, 471),
        (522,536),
        (541, len(df_centered))
    ]
    set_indeces_of_col_to_nan(df_centered, col_name, index_ranges)
    array_lag_info=lagged_data_correlation_multi_column(df_centered.iloc[:,2:25])
    #pol_1.func_visulized_df(df_centered, keys_df)
    df_help=df_centered.copy()
    print(df_help['Flotation Column 06 Air Flow_mean'].corr(df_help["Flotation Column 07 Air Flow_mean"]))
    print(df_help['% Iron Feed_mean'].corr(df_help["% Silica Feed_mean"]))
    print(df_help['Flotation Column 01 Air Flow_mean'].corr(df_help["Flotation Column 02 Air Flow_mean"]))
    print(df_help['Flotation Column 01 Air Flow_mean'].corr(df_help["Flotation Column 03 Air Flow_mean"]))
    print(df_help['Flotation Column 01 Level_mean'].corr(df_help['Flotation Column 02 Level_mean']))
    print(df_help['Flotation Column 01 Level_mean'].corr(df_help['Flotation Column 03 Level_mean']))
    print(df_help['Flotation Column 04 Level_mean'].corr(df_help['Flotation Column 05 Level_mean']))
    print(df_help['Flotation Column 06 Level_mean'].corr(df_help['Flotation Column 07 Level_mean']))
    print(df_help.keys()[2:25])
    #pol_1.vis_corrL(df_help.iloc[:,2:25])
    array_lag_info[-1][1]=0
    array_lag_info[-2][1] = 0
    array_lag_info[1][1] = 217
    df_lagged=lag_data_set(df_help,array_lag_info)
    for key in df_lagged.keys():
        if "Flotation Column 04 Air Flow_mean" in key or "Flotation Column 05 Air Flow_mean" in key:
            df_lagged =df_lagged.drop(key,axis=1)
    pol_1.vis_corrL(df_lagged.iloc[:, 2:25])
    cleaned_df = df_lagged.dropna()
    print("Remaining row indices:", cleaned_df.index.tolist())
    #df_lagged.to_csv("df_lagged" + now.strftime("%m_%d_%H_%M_%S") + '.csv', index=False)
    spectral_print(cleaned_df)

if __name__ == "__main__":
    #df = pd.read_csv("mining_process_means09_09_10_07_21.csv")
    #addaped_frequence_dates(df,True)
    #addaped_frequence_dates_data_originale(True)
    df = pd.read_csv("mining_process_dates_low_res09_26_09_21_07.csv")# downsampled data
    data_preteating_1(df)