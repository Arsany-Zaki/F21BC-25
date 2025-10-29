
import pandas as pd

file_path = "data/raw/Concrete_Data.csv"

df = pd.read_csv(file_path)
print(df.head())