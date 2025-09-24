import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import numpy as np

# Set seaborn style for better visuals
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display first few rows
    print("First 5 rows of the dataset:")
    print(df.head())
    
    # Check data types and missing values
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Clean dataset (fill missing values with mean for numerical columns)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    
except FileNotFoundError:
    print("Error: Dataset file not found.")
except Exception as e:
    print(f"Error occurred: {str(e)}")

# Task 2: Basic Data Analysis
try:
    # Basic statistics
    print("\nBasic Statistics:")
    print(df.describe())
    
    # Group by species and compute mean for numerical columns
    print("\nMean values by species:")
    group_means = df.groupby('species').mean()
    print(group_means)
    
    # Observations
    print("\nObservations:")
    print("- Setosa species has the smallest sepal length and width on average.")
    print("- Virginica species has the largest petal length and width on average.")
    print("- Versicolor falls between Setosa and Virginica in most measurements.")
    
except Exception as e:
    print(f"Error in analysis: {str(e)}")

# Task 3: Data Visualization
try:
    # 1. Line chart (simulating a trend by plotting mean measurements per species)
    plt.figure(figsize=(10, 6))
    for column in numerical_cols:
        plt.plot(group_means.index, group_means[column], marker='o', label=column)
    plt.title('Mean Measurements by Species')
    plt.xlabel('Species')
    plt.ylabel('Measurement (cm)')
    plt.legend()
    plt.savefig('line_chart.png')
    plt.close()
    
    # 2. Bar chart (average petal length per species)
    plt.figure(figsize=(8, 6))
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title('Average Petal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Petal Length (cm)')
    plt.savefig('bar_chart.png')
    plt.close()
    
    # 3. Histogram (distribution of sepal length)
    plt.figure(figsize=(8, 6))
    sns.histplot(df['sepal length (cm)'], bins=20, kde=True)
    plt.title('Distribution of Sepal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.savefig('histogram.png')
    plt.close()
    
    # 4. Scatter plot (sepal length vs petal length)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', size='species', data=df)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend(title='Species')
    plt.savefig('scatter_plot.png')
    plt.close()
    
    print("\nVisualizations created: line_chart.png, bar_chart.png, histogram.png, scatter_plot.png")
    
except Exception as e:
    print(f"Error in visualization: {str(e)}")