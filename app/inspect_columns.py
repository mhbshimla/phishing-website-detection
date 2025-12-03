import pandas as pd

df = pd.read_csv("phishing_data.csv")
columns = df.drop(columns=["index", "Result"], errors="ignore").columns.tolist()
print(columns)