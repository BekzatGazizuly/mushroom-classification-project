import pandas as pd

def load_data():
    df = pd.read_csv("mushrooms.csv", sep=";")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    print("\nTarget values:", df['class'].unique())
    return df

if __name__ == "__main__":
    load_data()
