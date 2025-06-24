import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Loading the Titanic Dataset
df = pd.read_csv('/content/Titanic-Dataset.csv')

# Displaying the first few rows
print(df.head())

# Summary Statistics
print("Summary Statistics")
print(df.describe(include='all'))

# Checking for missing values
print("Missing Values")
print(df.isnull().sum())

# Visualizing missing data
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Heatmap')
plt.show()

# Using Histogram for numerical columns
df.hist(bins=30, figsize=(12, 8), color='steelblue')
plt.suptitle("Histograms of Numeric Columns")
plt.show()

# Using Boxplot --> Age by Survived
sns.boxplot(data=df, x='Survived', y='Age')
plt.title("Boxplot of Age by Survival")
plt.show()

# Correlation Matrix
numeric_df = df.select_dtypes(include='number')  # Select only numeric columns
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot (subset of features)
sns.pairplot(df[['Survived', 'Age', 'Fare', 'Pclass']])
plt.suptitle("Pairplot of Selected Features", y=1.02)
plt.show()

# Detecting Skewness
print("Skewness")
print(df[['Age', 'Fare']].skew())

# For Multicollinearity Check 
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Select only numeric features and drop NaNs
df_vif = df[['Age', 'Fare', 'Pclass']].dropna()
vif_data = pd.DataFrame()
vif_data['feature'] = df_vif.columns
vif_data['VIF'] = [variance_inflation_factor(df_vif.values, i) for i in range(df_vif.shape[1])]
print("Variance Inflation Factor (VIF)")
print(vif_data)

# Interactive Plot with Plotly 
fig = px.histogram(df, x="Age", color="Survived", nbins=30, title="Age Distribution by Survival")
fig.show()
