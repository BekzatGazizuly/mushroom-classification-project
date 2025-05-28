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

    # Optional: use smaller subset for faster plot
    X_train_np = X_train.to_numpy()[:10000]
    y_train_np = y_train.to_numpy()[:10000]
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    depths = range(1, 16)  # limit to depth 1–15 for speed
    accuracies = []

    for depth in depths:
        tree = DecisionTree(max_depth=depth, min_samples_split=10, criterion="gini")
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
        print(f"Depth: {depth}, Accuracy: {acc:.4f}")

    plt.plot(depths, accuracies, marker='o')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Tree Depth')
    plt.grid(True)
    plt.savefig("accuracy_by_depth.png")
    plt.show()
