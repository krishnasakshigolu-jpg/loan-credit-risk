from pathlib import Path

import pandas as pd

df = pd.read_csv(Path(__file__).parent / "train.csv")
print(df.head())
print(df.isnull().sum())
print(df.shape)
