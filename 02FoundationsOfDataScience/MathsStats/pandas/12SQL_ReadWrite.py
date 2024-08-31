# $>> pip install sqlalchemy
import pandas as pd
import sqlalchemy

pd.options.display.width= None
pd.options.display.max_columns= None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# # Install MySQL: Open Workbench, Create user, connect via user, Create database, use database...
# # execute SQL Script from ./data/12SQL.xlsx file to create and populate 3 tables >>>>
# $>> pip install PyMySQL
# $>> pip install sqlalchemy

connection_string = 'mysql+pymysql://AnodiamUser:Anodiam123$@127.0.0.1:3306/AnodiamDb'
# 'mysql+pymysql://<USERID>:<PASSWORD>@<MY_SQL_SERVER>:<PORT>/<DATABASE>'
engine = sqlalchemy.create_engine(connection_string)

# df_customers = pd.read_sql_table('customers'.lower(), engine)
# df_customers = pd.read_sql_table('Customers'.lower(), engine, columns=['customer_name', 'phone_number'])
# print(df_customers)
# df_products = pd.read_sql_table('PrOdUcTs'.lower(), engine, columns=['product_name', 'price'])
# print(df_products)

# query = '''
# select customers.customer_name, customers.phone_number, products.product_name, products.price
# from orders, customers, products
# where orders.customer_id = customers.id and orders.product_id = products.id
# '''
# df_orders = pd.read_sql_query(query, engine)
# print(f'\ndf_orders.head(10) = \n{df_orders.head(10)}')
# print(f'\ndf_orders.tail(10) = \n{df_orders.tail(10)}')
# print(f'\ndf_orders.info() = \n{df_orders.info()}')
# print(f'\ndf_orders.describe() = \n{df_orders.describe()}')
# print(f'\ndf_orders.shape = \n{df_orders.shape}')
# print(f'\ndf_orders.columns = \n{df_orders.columns}')
# print(f'\ndf_orders.index = \n{df_orders.index}')
# print(f'\ndf_orders.dtypes = \n{df_orders.dtypes}')

# Large Chunks of data
# --------------------
# chunk_size = 16
# chunk_count = 0
# for chunk in pd.read_sql_query(query, engine, chunksize=chunk_size):
#     chunk_count += 1
#     print(f'------------------------\nchunk[{chunk_count}]\n------------------------\n{chunk}')

# Insert row from an Excel sheet
df_new_prods = pd.read_excel('./data/12SQL.xlsx', sheet_name='InsertProducts')
# print(df_new_prods)
df_new_prods = df_new_prods.rename(columns={'Product Name': 'product_name', 'Price': 'price'})
# print(df_new_prods)
df_new_prods.to_sql('products', con=engine, if_exists='append', index=False)
df_products = pd.read_sql_table('PrOdUcTs'.lower(), engine, columns=['product_name', 'price'])
print(df_products.product_name.count())
