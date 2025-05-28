class TreeNode:
    def __init__(self, is_leaf=False, prediction=None, feature_index=None, threshold=None):
        self.is_leaf = is_leaf
        self.prediction = prediction      # only if is_leaf
        self.feature_index = feature_index  # which feature to split on
        self.threshold = threshold          # value to compare against
        self.left = None
        self.right = None

    def test(self, x):
        return x[self.feature_index] <= self.threshold
