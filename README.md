# 🍄 Mushroom Tree Classifier

This project implements decision tree predictors from scratch for binary classification to determine whether mushrooms are poisonous or edible. The project also includes overfitting analysis, hyperparameter tuning, and an optional Random Forest extension.

## 📊 Dataset

- Based on the [UCI Mushroom Dataset](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
- Contains over 61,000 mushroom entries
- Categorical features transformed via one-hot encoding

## 🧠 Features

- ✅ Custom Decision Tree implementation
- ✅ Gini, Entropy, and Scaled Entropy splitting criteria
- ✅ Depth and min-sample stopping criteria
- ✅ Accuracy plots and comparisons
- ✅ Random Forest ensemble (from scratch)

## 📂 Project Structure

```
├── decision_tree.py         # Core decision tree logic
├── random_forest.py         # Random forest built on top of the tree
├── tree_node.py             # Tree node structure
├── load_data.py             # Load and view dataset
├── preprocess.py            # One-hot encoding and preprocessing
├── split_data.py            # Train-test splitting
├── run_tree.py              # Run decision trees with different criteria
├── test_random_forest.py    # Run random forest experiment
├── report.tex               # Full LaTeX report (10+ pages)
├── Mushrooms_report.pdf     # Final compiled report
├── *.py, *.png              # Supporting scripts and visualizations
```

## 📈 Visualizations

- `accuracy_by_depth.png`: Model performance vs tree depth
- `criteria_comparison.png`: Accuracy comparison of splitting criteria
- `comparison_all_methods.png`: Final comparison with Random Forest

## 📑 Report

A complete report is available:  
📄 **`Mushrooms_report.pdf`**

## 🔧 Requirements

- Python 3.7+
- `numpy`, `pandas`, `matplotlib`, `scikit-learn`

## 🧪 Run Example

```bash
python3 run_tree.py
python3 test_random_forest.py
```

## 🧾 License

MIT License (or follow course guidelines)