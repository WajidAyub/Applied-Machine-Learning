import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
file_path = "Car_sales.csv"
df = pd.read_csv(file_path)

# Show first rows
print(df.head())
print(df.info())
print(df.isnull().sum())

# Drop irrelevant text columns
df = df.drop(columns=["Manufacturer", "Model", "Latest_Launch"])

# Fill missing numeric values with mean
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

# Encode categorical variable
le = LabelEncoder()
df["Vehicle_type"] = le.fit_transform(df["Vehicle_type"])

# Define features (X) and target (y)
X = df.drop(columns=["Sales_in_thousands"])
y = df["Sales_in_thousands"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Predictions
y_pred_lin = lin_reg.predict(X_test)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_train)
y_pred_ridge = ridge.predict(X_test_scaled)

results = {
    "Linear MSE": mean_squared_error(y_test, y_pred_lin),
    "Linear R2": r2_score(y_test, y_pred_lin),
    "Ridge MSE": mean_squared_error(y_test, y_pred_ridge),
    "Ridge R2": r2_score(y_test, y_pred_ridge)
}
print("Model Comparison:\n", results)

feature_names = X.columns
lin_coef = lin_reg.coef_
ridge_coef = ridge.coef_

plt.figure(figsize=(12,6))
plt.bar(feature_names, lin_coef, alpha=0.6, label="Linear Regression")
plt.bar(feature_names, ridge_coef, alpha=0.6, label="Ridge Regression")
plt.xticks(rotation=75)
plt.ylabel("Coefficient Value")
plt.title("Feature Importance (Model Coefficients)")
plt.legend()
plt.tight_layout()
plt.savefig('ridge_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()


