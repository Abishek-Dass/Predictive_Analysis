
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("customer_data.csv") // load dataset

print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

// data preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Annual_Income'].fillna(df['Annual_Income'].median(), inplace=True)
df['Spending_Score'].fillna(df['Spending_Score'].mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

# Encode categorical columns
le_gender = LabelEncoder()
le_response = LabelEncoder()

df['Gender'] = le_gender.fit_transform(df['Gender'])   # Male=1, Female=0
df['Response'] = le_response.fit_transform(df['Response'])  # Yes=1, No=0

print("\nCleaned Data:")
print(df.head())



# Customer Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.show()

# Income vs Spending Score
plt.figure(figsize=(8,5))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Gender', data=df)
plt.title("Income vs Spending Score")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


X = df[['Age', 'Annual_Income', 'Purchase_Frequency']]
y = df['Spending_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("\n--- Linear Regression Results ---")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))


X_cls = df[['Age', 'Annual_Income', 'Purchase_Frequency', 'Spending_Score']]
y_cls = df['Response']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

log_model = LogisticRegression()
log_model.fit(X_train_c, y_train_c)

y_pred_c = log_model.predict(X_test_c)

print("\n--- Logistic Regression Results ---")
print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_c, y_pred_c))
print("\nClassification Report:")
print(classification_report(y_test_c, y_pred_c))


seg = df[['Annual_Income', 'Spending_Score']]

scaler = StandardScaler()
seg_scaled = scaler.fit_transform(seg)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(seg_scaled)

plt.figure(figsize=(8,5))
sns.scatterplot(x='Annual_Income', y='Spending_Score', hue='Cluster', palette='Set2', data=df)
plt.title("Customer Segmentation using K-Means")
plt.show()


print("\n--- Dashboard Metrics ---")
print("Total Customers:", len(df))
print("Average Income:", df['Annual_Income'].mean())
print("Average Spending Score:", df['Spending_Score'].mean())
print("Active Responders:", df['Response'].sum())
print("Customer Segments:\n", df['Cluster'].value_counts())

# Export cleaned dataset for Power BI dashboard
df.to_csv("customer_dashboard_data.csv", index=False)

print("\nDashboard dataset exported successfully as customer_dashboard_data.csv")
