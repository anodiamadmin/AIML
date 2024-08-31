import pandas as pd
import time

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Read the CSV file into a pandas DataFrame
df_users_age = pd.read_csv('./data/13HugeDatasetsPythonSpeed.csv')
print(f'df_users_age.shape = {df_users_age.shape}\ndf_users_age.head()\n{df_users_age.head()}')

# Calculate Time required to find avg age of all users using unoptimized data-frame
start_time = time.process_time() * 1000000
avg_age_unoptimized = df_users_age['Age'].mean()
end_time = time.process_time() * 1000000
time_req_unoptimized = end_time - start_time

# Calculate memory usage before optimization
initial_memory = df_users_age.memory_usage().sum()
print(f'\nMemory consumption by df_users_age = {initial_memory} bytes')

# Optimize memory usage
# Select only the required columns
# df_users_age = df_users_age[['UID', 'Sex', 'Age']]
df_users_age = pd.read_csv('./data/13HugeDatasetsPythonSpeed.csv',
                           usecols=['UID', 'Sex', 'Age'])
memory_three_cols = df_users_age.memory_usage().sum()
print(f'Memory consumption by df_users_age[[\'UID\', \'Sex\', \'Age\']] = '
      f'{memory_three_cols} bytes => {round((1-memory_three_cols/initial_memory)*100, 2)}% '
      f'memory saved.')

# Convert Int64 to Int8
df_users_age['Age'] = df_users_age['Age'].astype('int8')
memory_casted = df_users_age.memory_usage().sum()
print(f'Memory consumption by casting Age column from int64 to int8 = '
      f'{memory_casted} bytes => {round((1-memory_casted/memory_three_cols)*100, 2)}% memory saved.')

# Use categorical data type for 'Sex' column
df_users_age['Sex'] = df_users_age['Sex'].astype('category')
print(f'df_users_age.shape = {df_users_age.shape}\ndf_users_age.head()\n{df_users_age.head()}'
      f' df_users_age.dtypes\n{df_users_age.dtypes}')
memory_casted_category = df_users_age.memory_usage().sum()
print(f'Memory consumption by casting Sex column from object to category = '
      f'{memory_casted_category} bytes => '
      f'{round((1-memory_casted_category/memory_casted)*100, 2)}% memory saved.')

# Use One Hot Encoding for 'Sex' column
df_users_age = pd.get_dummies(df_users_age, columns=['Sex'])
df_users_age = df_users_age.drop(['Sex_Male'], axis=1)
print(f'df_users_age.shape = {df_users_age.shape}\ndf_users_age.head()\n{df_users_age.head()}'
      f' df_users_age.dtypes\n{df_users_age.dtypes}')
memory_one_hot_encoded = df_users_age.memory_usage().sum()
print(f'Memory consumption by One Hot Encoding Sex column = '
      f'{memory_one_hot_encoded} bytes => '
      f'{round((1-memory_one_hot_encoded/memory_casted_category)*100, 2)}% memory saved.')

# Calculate Time required to find avg age of all users using optimized data-frame
start_time2 = time.process_time() * 1000000
avg_age_optimized = df_users_age['Age'].mean()
end_time2 = time.process_time() * 1000000
time_req_optimized = end_time2 - start_time2

print(f'Time to find avg age using unoptimized data-frame = '
      f'{time_req_unoptimized} micro seconds :: Result = {avg_age_unoptimized}')
print(f'Time to find avg age using optimized data-frame = '
      f'{time_req_optimized} micro seconds :: Result = {avg_age_optimized}')
# print(f'Time saved by using optimized data-frame = '
#       f'{round((time_req_unoptimized-time_req_optimized)/time_req_unoptimized*100, 2)}%')
