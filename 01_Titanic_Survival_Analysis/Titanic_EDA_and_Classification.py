# Titanic Survival Analysis and Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, 
                            classification_report, mean_squared_error, 
                            r2_score, roc_curve, roc_auc_score)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("="*80)
print("Titanic Survival Analysis")
print("Dataset: Titanic Dataset (with Missing Values)")
print("="*80)


# ============================================================================
# TASK 1: Load Dataset with Missing Values
# ============================================================================
print("\n" + "="*80)
print("TASK 1: READING DATASET FROM CSV")
print("="*80)

# Load Titanic dataset (You can download from Kaggle or use seaborn's built-in)
# For this example, we'll use seaborn's titanic dataset which has missing values
df = sns.load_dataset('titanic')

# Save to CSV for reference
df.to_csv('titanic_dataset.csv', index=False)
print("\n✓ Dataset loaded successfully!")
print(f"✓ Dataset saved as 'titanic_dataset.csv'")


# ============================================================================
# TASK 2: Initial Data Exploration
# ============================================================================
print("\n" + "="*80)
print("TASK 2: EXPLORING THE DATASET")
print("="*80)

# Display first few rows
print("\n1. First 5 rows of the dataset:")
print(df.head())

# Display dataset shape
print(f"\n2. Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# Display basic information
print("\n3. Dataset Information:")
print(df.info())

# Display summary statistics
print("\n4. Summary Statistics for Numerical Columns:")
print(df.describe())

# Display column names and data types
print("\n5. Column Names and Data Types:")
for col in df.columns:
    print(f"   - {col}: {df[col].dtype}")


# ============================================================================
# TASK 3: CHECKING FOR MISSING VALUES
# ============================================================================
print("\n" + "="*80)
print("TASK 3: IDENTIFYING MISSING VALUES")
print("="*80)

# Check for missing values
missing_counts = df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\n1. Missing Values Count and Percentage:")
missing_df = pd.DataFrame({
    'Missing Count': missing_counts,
    'Percentage': missing_percentage.round(2)
})
print(missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False))

# Visualize missing values
plt.figure(figsize=(12, 6))
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
plt.bar(range(len(missing_data)), missing_data.values, color='coral')
plt.xticks(range(len(missing_data)), missing_data.index, rotation=45, ha='right')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.title('Missing Values by Column')
plt.tight_layout()
plt.savefig('missing_values_visualization.png', dpi=300, bbox_inches='tight')

# Display rows with missing values (sample)
print("\n2. Sample of Rows with Missing Values:")
print(df[df.isnull().any(axis=1)].head(10))


# ============================================================================
# TASK 4 & 5: DATA PREPROCESSING
# ============================================================================
df_processed = df.copy()
df_processed['age'] = df_processed['age'].fillna(df_processed['age'].median())
df_processed['embarked'] = df_processed['embarked'].fillna(df_processed['embarked'].mode()[0])
df_processed['fare'] = df_processed['fare'].fillna(df_processed['fare'].median())

# ============================================================================
# TASK 6: DATA VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("TASK 6: DATA VISUALIZATION")
print("="*80)

print("\n1. Creating various visualizations...")

# 1. Distribution of Age
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df_processed['age'], bins=30, color='steelblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')

plt.subplot(1, 2, 2)
plt.boxplot(df_processed['age'])
plt.ylabel('Age')
plt.title('Box Plot of Age')
plt.tight_layout()
plt.savefig('age_distribution.png', dpi=300, bbox_inches='tight')

# 2. Survival Rate by Gender
plt.figure(figsize=(10, 5))
survival_by_sex = df_processed.groupby('sex')['survived'].mean()
plt.bar(survival_by_sex.index, survival_by_sex.values, color=['lightcoral', 'lightblue'])
plt.xlabel('Gender')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Gender')
plt.ylim(0, 1)
for i, v in enumerate(survival_by_sex.values):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('survival_by_gender.png', dpi=300, bbox_inches='tight')

# 3. Survival Rate by Class
plt.figure(figsize=(10, 5))
survival_by_class = df_processed.groupby('pclass')['survived'].mean()
plt.bar(survival_by_class.index.astype(str), survival_by_class.values, 
        color=['gold', 'silver', 'brown'])
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.title('Survival Rate by Passenger Class')
plt.ylim(0, 1)
for i, v in enumerate(survival_by_class.values):
    plt.text(i, v + 0.02, f'{v:.2%}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('survival_by_class.png', dpi=300, bbox_inches='tight')

# 4. Fare Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df_processed['fare'], bins=50, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Distribution of Fare')

plt.subplot(1, 2, 2)
plt.hist(df_processed['fare'][df_processed['fare'] < 100], bins=30, 
         color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Fare (< 100)')
plt.ylabel('Frequency')
plt.title('Distribution of Fare (Filtered)')
plt.tight_layout()
plt.savefig('fare_distribution.png', dpi=300, bbox_inches='tight')

# 5. Scatter plot - Age vs Fare colored by Survival
plt.figure(figsize=(10, 6))
survived = df_processed['survived'] == 1
plt.scatter(df_processed[survived]['age'], df_processed[survived]['fare'], 
           alpha=0.5, c='green', label='Survived', s=30)
plt.scatter(df_processed[~survived]['age'], df_processed[~survived]['fare'], 
           alpha=0.5, c='red', label='Not Survived', s=30)
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Age vs Fare (Colored by Survival)')
plt.legend()
plt.tight_layout()
plt.savefig('age_vs_fare_scatter.png', dpi=300, bbox_inches='tight')

# 6. Count plot - Survival by Embarked Location
plt.figure(figsize=(10, 5))
sns.countplot(data=df_processed, x='embarked', hue='survived', palette='Set2')
plt.xlabel('Embarked Location')
plt.ylabel('Count')
plt.title('Survival Count by Embarked Location')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.tight_layout()
plt.savefig('survival_by_embarked.png', dpi=300, bbox_inches='tight')

print("✓ All visualizations created and saved!")

# ============================================================================
# TASK 7: TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("TASK 7: SPLITTING DATASET INTO TRAIN AND TEST SETS")
print("="*80)

# Prepare features for modeling
# Select relevant features and encode categorical variables
features_for_model = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
df_model = df_processed[features_for_model + ['survived']].copy()

# Encode categorical variables
label_encoders = {}
for col in ['sex', 'embarked']:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col])
    label_encoders[col] = le

# Define features and target
X = df_model.drop('survived', axis=1)
y = df_model['survived']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data split completed!")
print(f"   - Training set size: {X_train.shape[0]} samples ({(len(X_train)/len(X))*100:.1f}%)")
print(f"   - Testing set size: {X_test.shape[0]} samples ({(len(X_test)/len(X))*100:.1f}%)")
print(f"\n   Feature columns: {list(X.columns)}")

# ============================================================================
# TASK 8: TRAIN LOGISTIC REGRESSION MODEL
# ============================================================================
print("\n" + "="*80)
print("TASK 8: TRAINING LOGISTIC REGRESSION MODEL")
print("="*80)

# Note: The task asks for Linear Regression, but for binary classification
# (survived: 0 or 1), Logistic Regression is more appropriate
# We'll train both and show why Logistic Regression is better

print("\n1. Training Logistic Regression Model (Recommended for Classification):")
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
print("   ✓ Model trained successfully!")

# Make predictions
y_pred_train_log = log_model.predict(X_train)
y_pred_test_log = log_model.predict(X_test)

print("\n2. Also training Linear Regression (for comparison):")
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_train_lin = lin_model.predict(X_train)
y_pred_test_lin = lin_model.predict(X_test)
# Convert predictions to binary (threshold at 0.5)
y_pred_train_lin_binary = (y_pred_train_lin > 0.5).astype(int)
y_pred_test_lin_binary = (y_pred_test_lin > 0.5).astype(int)
print("   ✓ Model trained successfully!")

# ============================================================================
# TASK 9: MODEL EVALUATION
# ============================================================================
print("\n" + "="*80)
print("TASK 9: MODEL EVALUATION AND ACCURACY")
print("="*80)

# Evaluate Logistic Regression
print("\n" + "="*50)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)

train_accuracy_log = accuracy_score(y_train, y_pred_train_log)
test_accuracy_log = accuracy_score(y_test, y_pred_test_log)

print(f"\nTraining Accuracy: {train_accuracy_log:.4f} ({train_accuracy_log*100:.2f}%)")
print(f"Testing Accuracy: {test_accuracy_log:.4f} ({test_accuracy_log*100:.2f}%)")

# Confusion Matrix
cm_log = confusion_matrix(y_test, y_pred_test_log)
print("\nConfusion Matrix:")
print(cm_log)

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_test_log, 
                          target_names=['Not Survived', 'Survived']))

# Visualize Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')
plt.tight_layout()
plt.savefig('confusion_matrix_logistic.png', dpi=300, bbox_inches='tight')

# ROC Curve
y_pred_proba_log = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_log)
roc_auc = roc_auc_score(y_test, y_pred_proba_log)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')

print(f"\nROC AUC Score: {roc_auc:.4f}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': log_model.coef_[0]
}).sort_values('Coefficient', key=abs, ascending=False)

print("\nFeature Importance (Coefficients):")
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], 
         color=['green' if x > 0 else 'red' for x in feature_importance['Coefficient']])
plt.xlabel('Coefficient Value')
plt.title('Feature Importance in Logistic Regression Model')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

# Compare Training vs Testing Accuracy
print("\n" + "="*50)
print("LINEAR REGRESSION COMPARISON (For Reference)")
print("="*50)

train_accuracy_lin = accuracy_score(y_train, y_pred_train_lin_binary)
test_accuracy_lin = accuracy_score(y_test, y_pred_test_lin_binary)

print(f"\nTraining Accuracy: {train_accuracy_lin:.4f} ({train_accuracy_lin*100:.2f}%)")
print(f"Testing Accuracy: {test_accuracy_lin:.4f} ({test_accuracy_lin*100:.2f}%)")

# Visualize Model Comparison
plt.figure(figsize=(10, 6))
models = ['Logistic Regression', 'Linear Regression']
train_scores = [train_accuracy_log, train_accuracy_lin]
test_scores = [test_accuracy_log, test_accuracy_lin]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, train_scores, width, label='Training Accuracy', color='skyblue')
plt.bar(x + width/2, test_scores, width, label='Testing Accuracy', color='lightcoral')

plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Model Comparison: Training vs Testing Accuracy')
plt.xticks(x, models)
plt.ylim([0, 1])
plt.legend()
plt.grid(axis='y', alpha=0.3)

for i, (train, test) in enumerate(zip(train_scores, test_scores)):
    plt.text(i - width/2, train + 0.02, f'{train:.3f}', ha='center', fontweight='bold')
    plt.text(i + width/2, test + 0.02, f'{test:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')


