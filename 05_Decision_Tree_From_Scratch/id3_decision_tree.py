# ============================================================
# ID3 Decision Tree From Scratch
# Dataset: Social Network Ads
# ============================================================

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 1. LOAD / SYNTHESIZE DATASET
# ------------------------------------------------------------
# The original dataset 'Social_Network_Ads.csv' is generated synthetically 
# here to ensure the script runs standalone without external dependencies.
np.random.seed(42)
n_samples = 400
genders = np.random.choice(['Male', 'Female'], n_samples)
ages = np.random.randint(18, 60, n_samples)
salaries = np.random.randint(15000, 150000, n_samples)

# Simple logic for purchasing: Older and higher salary -> more likely to purchase
purchase_prob = (ages / 60) * 0.4 + (salaries / 150000) * 0.6
purchased = (purchase_prob > 0.5).astype(int)

df = pd.DataFrame({
    'Gender': genders,
    'Age': ages,
    'EstimatedSalary': salaries,
    'Purchased': purchased
})

print("First 5 rows:")
print(df.head())
print("\nColumns:", df.columns)

# ------------------------------------------------------------
# 2. PREPROCESSING – MAKE FEATURES CATEGORICAL
# ------------------------------------------------------------

# We only need Gender, Age, EstimatedSalary, Purchased
df = df[["Gender", "Age", "EstimatedSalary", "Purchased"]].copy()

# Bin Age into categories
df["AgeGroup"] = pd.cut(
    df["Age"],
    bins=[18, 30, 40, 60],
    labels=["Young", "Middle", "Senior"],
    include_lowest=True
)

# Bin EstimatedSalary into categories (tweak thresholds if needed)
df["SalaryGroup"] = pd.cut(
    df["EstimatedSalary"],
    bins=[0, 50000, 100000, 200000],
    labels=["Low", "Medium", "High"],
    include_lowest=True
)

# Convert to category dtype
df["Gender"] = df["Gender"].astype("category")
df["Purchased"] = df["Purchased"].astype("category")
df["AgeGroup"] = df["AgeGroup"].astype("category")
df["SalaryGroup"] = df["SalaryGroup"].astype("category")

# Drop original numeric columns (keep only categorical)
df_model = df[["Gender", "AgeGroup", "SalaryGroup", "Purchased"]].copy()

print("\nAfter preprocessing:")
print(df_model.head())
print("\nValue counts for AgeGroup:\n", df_model["AgeGroup"].value_counts())
print("\nValue counts for SalaryGroup:\n", df_model["SalaryGroup"].value_counts())

# ------------------------------------------------------------
# 3. TRAIN–TEST SPLIT
# ------------------------------------------------------------
train_df, test_df = train_test_split(df_model, test_size=0.3, random_state=0, stratify=df_model["Purchased"])

features = ["Gender", "AgeGroup", "SalaryGroup"]
target = "Purchased"

# Global majority class (fallback for unseen branches)
majority_class = train_df[target].mode()[0]

# ------------------------------------------------------------
# 4. ID3 IMPLEMENTATION
# ------------------------------------------------------------

def entropy(column):
    """Calculate entropy of a pandas Series (categorical)."""
    values, counts = np.unique(column, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))


def information_gain(data, feature, target):
    """Information gain of splitting 'data' on 'feature' w.r.t 'target'."""
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = 0.0
    for v, c in zip(values, counts):
        subset = data[data[feature] == v]
        weighted_entropy += (c / len(data)) * entropy(subset[target])

    return total_entropy - weighted_entropy


def id3(data, features, target):
    """Recursive ID3 algorithm. Returns a nested dict representing the tree."""
    # If all samples same class → leaf
    unique_classes = np.unique(data[target])
    if len(unique_classes) == 1:
        return unique_classes[0]

    # If no features left → majority class
    if len(features) == 0:
        return data[target].mode()[0]

    # Choose best feature by information gain
    gains = {f: information_gain(data, f, target) for f in features}
    best_feature = max(gains, key=gains.get)

    tree = {best_feature: {}}

    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value]

        if subset.empty:
            # No samples → majority class of current node
            tree[best_feature][value] = data[target].mode()[0]
        else:
            new_features = [f for f in features if f != best_feature]
            subtree = id3(subset, new_features, target)
            tree[best_feature][value] = subtree

    return tree


# Build the decision tree on training data
tree = id3(train_df, features, target)

print("\nConstructed Decision Tree (nested dict):")
print(tree)

# ------------------------------------------------------------
# 5. PREDICTION FUNCTION
# ------------------------------------------------------------

def predict_one(tree, sample, default_class=majority_class):
    """
    Predict class for a single sample (row) using the ID3 tree.
    tree: nested dict
    sample: pandas Series
    default_class: used if branch not found
    """
    if not isinstance(tree, dict):
        return tree

    # Root feature = first key
    feature = next(iter(tree))
    branches = tree[feature]
    value = sample[feature]

    # If branch exists, go deeper
    if value in branches:
        result = branches[value]
        if isinstance(result, dict):
            return predict_one(result, sample, default_class)
        else:
            return result
    else:
        # Unseen category during training
        return default_class


def predict(tree, data):
    return data.apply(lambda row: predict_one(tree, row), axis=1)


# ------------------------------------------------------------
# 6. EVALUATE TRAINING AND TEST ACCURACY
# ------------------------------------------------------------

train_pred = predict(tree, train_df)
test_pred = predict(tree, test_df)

train_acc = (train_pred == train_df[target]).mean()
test_acc = (test_pred == test_df[target]).mean()

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")
