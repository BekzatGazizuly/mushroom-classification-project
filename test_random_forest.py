import numpy as np
from load_data import load_data
from preprocess import preprocess
from split_data import split_data
from random_forest import RandomForest

if __name__ == "__main__":
    df = load_data()
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # convert to numpy
    X_train_np = X_train.to_numpy()
    y_train_np = y_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_test_np = y_test.to_numpy()

    forest = RandomForest(n_trees=10, max_depth=15, min_samples_split=10, criterion="gini")
    forest.fit(X_train_np, y_train_np)
    y_pred = forest.predict(X_test_np)

    accuracy = np.mean(y_pred == y_test_np)
    print("Random Forest Test Accuracy:", accuracy)
