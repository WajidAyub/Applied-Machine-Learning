# Decision Tree Classifier (From Scratch)

## 📌 Overview
This project showcases the implementation of the **ID3 Decision Tree Algorithm** using pure Python and NumPy. Building fundamental Machine Learning algorithms from scratch demonstrates a deep mathematical understanding of how models learn, stepping beyond the abstraction of high-level libraries like `scikit-learn`.

## 🧮 Mathematical Implementation
The decision tree recursively partitions the data based on **Information Theory**. Two core mathematical concepts were implemented manually:

### 1. Shannon Entropy
Calculates the impurity or randomness of a given dataset node.
$$ Entropy(S) = - \sum_{i=1}^{c} p_i \log_2(p_i) $$

### 2. Information Gain
Determines the best feature to split on by calculating the reduction in entropy after a dataset is split on an attribute.
$$ Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v) $$

## 🛠️ The ID3 Algorithm
The recursive `id3()` function was engineered to construct the tree structure:
1. Calculates the Information Gain for all available features.
2. Selects the feature with the highest gain as the root/decision node.
3. Splits the dataset into subsets based on the unique values of that feature.
4. Recursively calls itself on the subsets until a leaf node (pure class) is reached or no features remain.
5. Handles unseen data during prediction by falling back to the global majority class.

## 📊 Results
The algorithm successfully constructs a nested dictionary representation of the decision rules. When tested on a synthetic dataset consisting of `Age`, `Gender`, and `EstimatedSalary`:

- **Training Accuracy:** ~93.5%
- **Testing Accuracy:** ~88.3%

*(Note: The tree structure directly maps to logical `IF-ELSE` rules, making this model perfectly interpretable).*

## 💻 How to Run
1. Ensure you have `numpy` and `pandas` installed.
2. Run the script:
   ```bash
   python id3_decision_tree.py
   ```
3. The script will dynamically synthesize the Social Network Ads dataset, preprocess the numeric values into categorical bins, train the pure-Python model, output the nested dictionary tree structure, and print its accuracy.

---
*This project is part of my professional Machine Learning Engineering portfolio.*
