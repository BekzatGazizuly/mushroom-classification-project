import numpy as np
from load_data import load_data
from preprocess import preprocess
from split_data import split_data
from decision_tree import DecisionTree

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # convert to numpy
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    def predict_tree(tree, x):
        node = tree.root
        while not node.is_leaf:
            if node.test(x):
                node = node.left
            else:
                node = node.right
        return node.prediction

    # Gini
    tree_gini = DecisionTree(max_depth=15, min_samples_split=10, criterion="gini")
    tree_gini.fit(X_train_np, y_train_np)
    y_pred_gini = np.array([predict_tree(tree_gini, x) for x in X_test_np])
    print("Gini accuracy:", np.mean(y_pred_gini == y_test_np))

    # Entropy
    tree_entropy = DecisionTree(max_depth=15, min_samples_split=10, criterion="entropy")
    tree_entropy.fit(X_train_np, y_train_np)
    y_pred_entropy = np.array([predict_tree(tree_entropy, x) for x in X_test_np])
    print("Entropy accuracy:", np.mean(y_pred_entropy == y_test_np))

    # Scaled Entropy
    tree_scaled = DecisionTree(max_depth=15, min_samples_split=10, criterion="scaled_entropy")
    tree_scaled.fit(X_train_np, y_train_np)
    y_pred_scaled = np.array([predict_tree(tree_scaled, x) for x in X_test_np])
    print("Scaled Entropy accuracy:", np.mean(y_pred_scaled == y_test_np))