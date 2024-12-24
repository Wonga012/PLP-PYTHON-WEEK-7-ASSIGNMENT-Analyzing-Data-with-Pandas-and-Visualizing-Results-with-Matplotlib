# Task 1: Load and Explore the Dataset
# Import Libraries:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load the Dataset:
# Load the Iris dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
iris_data = pd.read_csv(url, header=None, names=column_names)

# 3. Display the First Few Rows:
print(iris_data.head())

# 4.Explore the Structure:
# Check data types and missing values
print(iris_data.info())
print(iris_data.isnull().sum())

# 5. Clean the Dataset:

# Drop missing values if any (not applicable for the Iris dataset)
iris_data.dropna(inplace=True)

# Task 2: Basic Data Analysis
# Compute Basic Statistics of numerical columns
print(iris_data.describe())

# 2. Group by Species and Compute Mean:
# Grouping by species and computing the mean of numerical columns
species_mean = iris_data.groupby('species').mean()
print(species_mean)

# 3. Identify Patterns:
# Observations
print("Observations:")
print("1. Setosa has the smallest petal length and width.")
print("2. Versicolor and Virginica have similar sepal lengths.")

# Task 3: Data Visualization
# 1. Line Chart:
# (Assuming you have time-series data; for this dataset, we can visualize trends in petal length across species.)
plt.figure(figsize=(10, 6))
sns.lineplot(data=iris_data, x='species', y='petal_length', marker='o')
plt.title('Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# 2. Bar Chart:

plt.figure(figsize=(10, 6))
sns.barplot(data=species_mean.reset_index(), x='species', y='petal_length')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

# 3. Histogram:

plt.figure(figsize=(10, 6))
sns.histplot(iris_data['sepal_length'], bins=10, kde=True)
plt.title('Distribution of Sepal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Frequency')
plt.show()

# 4. Scatter Plot:

plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_data, x='sepal_length', y='petal_length', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# 5. Additional Instructions
# Error Handling: Wrap data loading in a try-except block to handle errors.

try:
    iris_data = pd.read_csv(url, header=None, names=column_names)
except FileNotFoundError:
    print("Error: The file was not found.")
except pd.errors.EmptyDataError:
    print("Error: The file is empty.")
except Exception as e:
    print(f"An error occurred: {e}")







