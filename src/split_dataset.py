from sklearn.model_selection import train_test_split
import pandas as pd
import os

def main():
    df = pd.read_csv("data/processed/dataset.csv")
    train_and_validation, test = train_test_split(
        df,
        test_size=0.10,
        stratify= df["label"],
        random_state=42
    )
    split_value = 0.1/0.9
    train, validation = train_test_split(
        train_and_validation,
        test_size=split_value,
        stratify=train_and_validation["label"],
        random_state=42
    )
    os.makedirs("data/splits",exist_ok=True)
    test.to_csv("data/splits/test.csv", index=False)
    train.to_csv("data/splits/train.csv", index=False)
    validation.to_csv("data/splits/validation.csv", index=False)

if __name__ == "__main__":
    main()