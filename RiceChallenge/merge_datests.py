import pandas as pd

df1 = pd.read_csv("rice_test.csv", sep=",").round(10)
df2 = pd.read_csv("Rice_Cammeo_Osmancik.csv", sep=",").round(10)
df1=pd.DataFrame(df1)
df2=pd.DataFrame(df2)

print(df1)
print(df2)

merge_columns = [col for col in df1.columns]
print("keys:", merge_columns)

merged = df1.merge(df2, on=merge_columns, how="inner")
print(merged)

merged.to_csv("rice_test_with_class.csv", index=False)
#
