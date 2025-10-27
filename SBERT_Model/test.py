import pandas as pd

df = pd.read_csv(r"C:\Users\sumat\OneDrive\Desktop\Project\test.csv", encoding='latin1')
print(df.columns.tolist())
