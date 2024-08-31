import pandas as pd
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
# Reading 2 layers of header (header=[0, 1]) only possible for read_excel, not possible for red_csv
df_stock = pd.read_excel('./data/10stackUnstack.xlsx', sheet_name='10stackUnstack', header=[0, 1])
print(df_stock)

df_stock = df_stock.stack(level=0, dropna=True)
# df_stock = df_stock.stack(level=1, dropna=True)
print(df_stock)

df_stock = df_stock.unstack()
print(df_stock)
