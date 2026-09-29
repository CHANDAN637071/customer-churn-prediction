import pandas as pd

def load_and_clean(path="data/raw/telco_churn.csv"):
    df = pd.read_csv(path)

    # TotalCharges has blank strings (new customers with tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Target: Yes/No -> 1/0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # customerID is an identifier, not a feature
    df = df.drop(columns=["customerID"])
    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.shape)
    print(df["Churn"].value_counts(normalize=True))
    print(df.dtypes)
    df.to_csv("data/processed/clean.csv", index=False)