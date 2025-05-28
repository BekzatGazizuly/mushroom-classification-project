import numpy as np
from tree_node import TreeNode

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        # stopping condition
        if len(set(y)) == 1 or depth >= self.max_depth or len(y) < self.min_samples_split:
            prediction = int(np.round(np.mean(y)))
            return TreeNode(is_leaf=True, prediction=prediction)

        best_feat, best_thresh = self._best_split(X, y)
        if best_feat is None:
            prediction = int(np.round(np.mean(y)))
            return TreeNode(is_leaf=True, prediction=prediction)

        node = TreeNode(is_leaf=False, feature_index=best_feat, threshold=best_thresh)

        left_idx = X[:, best_feat] <= best_thresh
        right_idx = ~left_idx

        node.left = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        node.right = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return node

    def _best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_score = float('inf')

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left = y[X[:, feature_index] <= threshold]
                right = y[X[:, feature_index] > threshold]

                if len(left) == 0 or len(right) == 0:
                    continue

                if self.criterion == "gini":
                    score = self._gini_impurity(left, right)
                elif self.criterion == "entropy":
                    score = self._entropy_impurity(left, right)
                elif self.criterion == "scaled_entropy":
                    score = self._scaled_entropy(left, right)
                else:
                    raise ValueError("Unknown criterion")

                if score < best_score:
                    best_score = score
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold


    def _gini_impurity(self, left, right):
        def gini(labels):
            classes = np.unique(labels)
            impurity = 1.0
            for c in classes:
                p = np.sum(labels == c) / len(labels)
                impurity -= p ** 2
            return impurity

        n = len(left) + len(right)
        return (len(left) / n) * gini(left) + (len(right) / n) * gini(right)
  
    def _entropy_impurity(self, left, right):
        def entropy(labels):
            classes, counts = np.unique(labels, return_counts=True)
            probs = counts / counts.sum()
            return -np.sum(probs * np.log2(probs + 1e-9))  # avoid log(0)

        n = len(left) + len(right)
        return (len(left) / n) * entropy(left) + (len(right) / n) * entropy(right)

    def _scaled_entropy(self, left, right):
        def entropy(labels):
            counts = np.bincount(labels)
            probs = counts / len(labels)
            return -np.sum([p * np.log2(p) for p in probs if p > 0])

        total = len(left) + len(right)
        return (len(left) / total) * entropy(left) + (len(right) / total) * entropy(right)
