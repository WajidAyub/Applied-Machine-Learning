# Applied Machine Learning Portfolio

Welcome to my **Applied Machine Learning Portfolio**. This repository showcases a comprehensive collection of end-to-end Machine Learning pipelines, ranging from robust Exploratory Data Analysis (EDA) to advanced algorithmic implementation, evaluation, and MLOps deployment.

All projects in this repository are written in clean, modular Python and are designed to solve real-world classification and regression problems using industry-standard libraries like `scikit-learn`, `pandas`, `numpy`, and `matplotlib`.

---

## 📂 Project Directory

| Project | Description | Key Technologies & Skills |
| :--- | :--- | :--- |
| **[01_Titanic_Survival_Analysis](./01_Titanic_Survival_Analysis)** | An end-to-end binary classification pipeline predicting passenger survival. Highlights extensive EDA, handling missing data, and baseline Logistic Regression modeling. | `EDA`, `Data Imputation`, `Logistic Regression` |
| **[02_Telecom_Churn_Prediction](./02_Telecom_Churn_Prediction)** | Features two distinct pipelines: A Logistic Regression classifier utilizing **PCA** for dimensionality reduction, and a Ridge Regression model exploring feature scaling. | `PCA`, `Ridge Regression`, `Feature Scaling` |
| **[03_Heart_Disease_Prediction](./03_Heart_Disease_Prediction)** | A highly structured predictive pipeline engineered utilizing `scikit-learn`'s `Pipeline` and `ColumnTransformer` to enforce strict data transformations and prevent data leakage. | `Pipelines`, `ColumnTransformer`, `GridSearchCV` |
| **[04_Pulsar_Star_Classification](./04_Pulsar_Star_Classification)** | Focuses on rigorous model evaluation on highly imbalanced astronomical data. Uses Stratified K-Fold Cross Validation, Learning Curves, and statistical T-Tests. | `Stratified K-Fold`, `Learning Curves`, `T-Tests` |
| **[05_Decision_Tree_From_Scratch](./05_Decision_Tree_From_Scratch)** | A custom, pure-Python implementation of the **ID3 Decision Tree Algorithm** built from mathematical scratch utilizing Shannon Entropy and Information Gain. | `NumPy`, `Information Theory`, `Algorithm Design` |
| **[06_Breast_Cancer_Classification](./06_Breast_Cancer_Classification)** | Compares the diagnostic capabilities of an advanced Ensemble model (Random Forest) against a baseline linear model. Programmatically extracts and plots feature importances. | `Ensemble Models`, `Random Forest`, `Diagnostics` |
| **[07_Cardiovascular_Disease_Prediction](./07_Cardiovascular_Disease_Prediction)** | Analyzes 70,000 patient records. Involves deep data cleaning (removing physiological outliers), calculating BMI, and exhaustively tuning Gradient Boosting models. | `Data Cleaning`, `Gradient Boosting`, `GridSearchCV` |
| **[08_Breast_Cancer_Deployment_Pipeline](./08_Breast_Cancer_Deployment_Pipeline)** | An **MLOps** focused project. Implements **SMOTE** for severe class imbalances, evaluates 6 architectures (including SVMs and Neural Networks), and serializes the final models (`.pkl`) via `joblib` for web deployment. | `SMOTE`, `MLOps`, `Model Serialization (joblib)` |

---

## 🛠️ Technology Stack

- **Core Data Science:** Python, NumPy, Pandas
- **Machine Learning Algorithms:** Scikit-Learn (Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, SVM, KNN, Gaussian Naive Bayes, MLP Neural Networks)
- **Data Engineering & Imbalance:** Imbalanced-Learn (SMOTE), `ColumnTransformer`, `Pipeline`, `StandardScaler`
- **Model Tuning & Evaluation:** `GridSearchCV`, Stratified K-Fold CV, ROC-AUC, Precision/Recall optimization.
- **Data Visualization:** Matplotlib, Seaborn

## 🚀 How to Use
Each project directory is fully self-contained. Navigate to any project folder and run its respective `.py` script to automatically preprocess the datasets, train the models, and generate the diagnostic visualizations.

```bash
cd 08_Breast_Cancer_Deployment_Pipeline
python breast_cancer_deployment_pipeline.py
```

## 📬 Contact
Feel free to explore the code, and if you have any questions or opportunities, you can reach out to me via my GitHub profile.
