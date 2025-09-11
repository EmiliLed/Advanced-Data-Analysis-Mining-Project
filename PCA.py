import pandas as pd
import numpy as np

columns = [
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


df = pd.read_csv('MiningProcess_Flotation_Plant_Database.csv', usecols=columns, parse_dates=['date'])
df.iloc[:, 1:] = df.iloc[:, 1:].replace(',', '.', regex=True).astype(float)
print(df)

# Check for missing values
missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)
# Drop rows with missing values
df = df.dropna()


#Center and scale the data
from sklearn.preprocessing import StandardScaler

features = df.columns[1:]  # Exclude the date column
x = df.loc[:, features].values
x = StandardScaler().fit_transform(x)  # scaled_x=(x - x.mean()) / x.std()
scaled_x = pd.DataFrame(x, columns=df.columns[1:])
print(x)

# Perform PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=len(features))
principalComponents = pca.fit_transform(scaled_x)
loadings = pca.components_.T
scores = principalComponents
explained = pca.explained_variance_ratio_ * 100  # Percentage of variance explained by each component

principalDf = pd.DataFrame(data=principalComponents, columns=[f'PC{i+1}' for i in range(principalComponents.shape[1])])
print(principalDf)
# Explained variance
explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio of each principal component:")
print(explained_variance)
# Cumulative explained variance
cumulative_explained_variance = explained_variance.cumsum()
print("Cumulative explained variance:")
print(cumulative_explained_variance)
components=np.arange(1, len(cumulative_explained_variance) + 1)
# Plot explained variance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.plot( components,cumulative_explained_variance, label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

i=0
plt.figure(figsize=(8, 6))

# Plot Scores (the projected data)
plt.scatter(scores[:, i], scores[:, i+1], alpha=0.5)

# Plot Loadings (feature vectors)
for j, varname in enumerate(df.columns[1:]):
    plt.arrow(0, 0, loadings[j, i]*max(scores[:, i]), loadings[j, i+1]*max(scores[:, i+1]),
              color='r', alpha=0.5)
    plt.text(loadings[j, i]*max(scores[:, i])*1.15,
             loadings[j, i+1]*max(scores[:, i+1])*1.15,
             varname, color='black', ha='center', va='center')

plt.xlabel(f'PC {i+1} R²: {round(explained[i])} [%]')
plt.ylabel(f'PC {i+2} R²: {round(explained[i+1])} [%]')
plt.grid()
plt.tight_layout()
plt.show()

#Biplot
def biplot(scores, loadings, feature_names, pc1=0, pc2=1):
    plt.figure(figsize=(10, 7))
    plt.scatter(scores[:, pc1], scores[:, pc2], alpha=0.5)

    for i, feature in enumerate(feature_names):
        plt.arrow(0, 0, loadings[i, pc1]*max(scores[:, pc1]), loadings[i, pc2]*max(scores[:, pc2]),
                  color='r', alpha=0.5)
        plt.text(loadings[i, pc1]*max(scores[:, pc1])*1.15,
                 loadings[i, pc2]*max(scores[:, pc2])*1.15,
                 feature, color='black', ha='center', va='center')

    plt.xlabel(f'PC {pc1+1} R²: {round(explained[pc1])} [%]')
    plt.ylabel(f'PC {pc2+1} R²: {round(explained[pc2])} [%]')
    plt.grid()
    plt.title('PCA Biplot')
    plt.tight_layout()
    plt.show()

biplot(scores, loadings, df.columns[1:], pc1=0, pc2=1)
biplot(scores, loadings, df.columns[1:], pc1=1, pc2=2)
biplot(scores, loadings, df.columns[1:], pc1=2, pc2=3)
biplot(scores, loadings, df.columns[1:], pc1=3, pc2=4)
biplot(scores, loadings, df.columns[1:], pc1=4, pc2=5)


#Compute control chart limits
def compute_control_limits(scores, n_std=3):
    mean = np.mean(scores, axis=0)
    std_dev = np.std(scores, axis=0)
    upper_limit = mean + n_std * std_dev
    lower_limit = mean - n_std * std_dev
    return mean, upper_limit, lower_limit

# Plot control charts for the first few principal components
for i in range(5):  # Plot for the first 5 principal components
    mean, upper_limit, lower_limit = compute_control_limits(scores[:, i])
    plt.figure(figsize=(12, 6))
    plt.plot(scores[:, i], marker='o', linestyle='-', label=f'PC {i+1} Scores')
    plt.axhline(mean, color='green', linestyle='--', label='Mean')
    plt.axhline(upper_limit, color='red', linestyle='--', label='Upper Control Limit')
    plt.axhline(lower_limit, color='red', linestyle='--', label='Lower Control Limit')
    plt.title(f'Control Chart for Principal Component {i+1}')
    plt.xlabel('Sample Index')
    plt.ylabel('Score Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


#Increase the number of principal components until you reach 95% of explained variance
n_components_95 = np.argmax(cumulative_explained_variance >= 0.95) + 1
print(f'Number of principal components to reach 95% explained variance: {n_components_95}')
#Re-run PCA with the optimal number of components
pca_optimal = PCA(n_components=n_components_95)
principalComponents_optimal = pca_optimal.fit_transform(scaled_x)
explained_optimal = pca_optimal.explained_variance_ratio_ * 100  # Percentage of variance explained by each component
print("Explained variance ratio of each principal component (optimal):")
print(explained_optimal)
# Cumulative explained variance
cumulative_explained_variance_optimal = explained_optimal.cumsum()
print("Cumulative explained variance (optimal):")
print(cumulative_explained_variance_optimal)
components_optimal=np.arange(1, len(cumulative_explained_variance_optimal) + 1)
# Plot explained variance
plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_optimal) + 1), explained_optimal, alpha=0.5, align='center', label='Individual explained variance')
plt.plot( components_optimal,cumulative_explained_variance_optimal, label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

#Scores and loadings for optimal PCA
loadings_optimal = pca_optimal.components_.T
scores_optimal = principalComponents_optimal
#Biplot for optimal PCA
biplot(scores_optimal, loadings_optimal, df.columns[1:], pc1=0, pc2=1)
biplot(scores_optimal, loadings_optimal, df.columns[1:], pc1=1, pc2=2)
biplot(scores_optimal, loadings_optimal, df.columns[1:], pc1=2, pc2=3)
if n_components_95>3:
    biplot(scores_optimal, loadings_optimal, df.columns[1:], pc1=3, pc2=4)
if n_components_95>4:
    biplot(scores_optimal, loadings_optimal, df.columns[1:], pc1=4, pc2=5)
# Plot control charts for the optimal principal components
for i in range(n_components_95):  # Plot for the optimal number of principal components
    mean, upper_limit, lower_limit = compute_control_limits(scores_optimal[:, i])
    plt.figure(figsize=(12, 6))
    plt.plot(scores_optimal[:, i], marker='o', linestyle='-', label=f'PC {i+1} Scores')
    plt.axhline(mean, color='green', linestyle='--', label='Mean')
    plt.axhline(upper_limit, color='red', linestyle='--', label='Upper Control Limit')
    plt.axhline(lower_limit, color='red', linestyle='--', label='Lower Control Limit')
    plt.title(f'Control Chart for Principal Component {i+1} (Optimal)')
    plt.xlabel('Sample Index')
    plt.ylabel('Score Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# Reconstruct the original data from the optimal PCA
reconstructed_data = pca_optimal.inverse_transform(principalComponents_optimal)
reconstructed_df = pd.DataFrame(reconstructed_data, columns=scaled_x.columns)
# Compare original and reconstructed data
for column in scaled_x.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(scaled_x[column], label='Original', alpha=0.7)
    plt.plot(reconstructed_df[column], label='Reconstructed', alpha=0.7)
    plt.title(f'Original vs Reconstructed Data for {column}')
    plt.xlabel('Sample Index')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
# Calculate reconstruction error
reconstruction_error = np.mean((scaled_x - reconstructed_df) ** 2)
print(f'Mean Squared Reconstruction Error: {reconstruction_error}')
