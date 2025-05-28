import pandas as pd

def preprocess(df):
    # Convert class labels: 'p' → 1 (poisonous), 'e' → 0 (edible)
    df['class'] = df['class'].map({'p': 1, 'e': 0})

    # Separate features and target
    X = df.drop(columns=['class'])
    y = df['class']

    # One-hot encode categorical features
    X_encoded = pd.get_dummies(X)

    return X_encoded, y

if __name__ == "__main__":
    from load_data import load_data
    df = load_data()
    X, y = preprocess(df)
    print("Shape of X:", X.shape)
    print("Target distribution:\n", y.value_counts())
