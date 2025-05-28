import numpy as np
import matplotlib.pyplot as plt
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

    criteria = ["gini", "entropy", "scaled_entropy"]
    accuracies = []

    for crit in criteria:
        tree = DecisionTree(max_depth=15, min_samples_split=10, criterion=crit)
        tree.fit(X_train_np, y_train_np)

        def predict_tree(tree, x):
            node = tree.root
            while not node.is_leaf:
                if node.test(x):
                    node = node.left
                else:
                    node = node.right
            return node.prediction

        y_pred = np.array([predict_tree(tree, x) for x in X_test_np])
        acc = np.mean(y_pred == y_test_np)
        accuracies.append(acc)
        print(f"{crit} accuracy: {acc:.4f}")

    # Plot
    plt.figure()
    plt.bar(criteria, accuracies, color=['blue', 'green', 'orange'])
    plt.ylabel('Test Accuracy')
    plt.title('Comparison of Splitting Criteria')
    plt.ylim(0, 1)
    plt.savefig("criteria_comparison.png")
    plt.show()
