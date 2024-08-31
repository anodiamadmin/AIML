import pandas as pd

# Create a DataFrame with "carbonated?", "temperature", "sugar(tsp.)", and "calories" columns:
df_soft_drink = pd.DataFrame([
  {'drink': 'soda', 'carbonated?': True, 'temperature': 'cold', 'sugar(tsp)': 10.5, 'calories': 150},
  {'drink': 'coffee', 'carbonated?': False, 'temperature': 'hot', 'sugar(tsp)': 3, 'calories': 31},
  {'drink': 'smoothie', 'carbonated?': False, 'temperature': 'cold', 'sugar(tsp)': 6, 'calories': 85},
  {'drink': 'water', 'carbonated?': False, 'temperature': 'cold', 'sugar(tsp)': 0, 'calories': 0},
  {'drink': 'tea', 'carbonated?': False, 'temperature': 'hot', 'sugar(tsp)': 2, 'calories': 43},
  {'drink': 'lemonade', 'carbonated?': False, 'temperature': 'cold', 'sugar(tsp)': 9.5, 'calories': 125},
  {'drink': 'slushy', 'carbonated?': False, 'temperature': 'cold', 'sugar(tsp)': 8, 'calories': 99},
])
# print(f'Dataframe:\n{df_soft_drink}')

# LOC:: Set the row label indexes to the "drink" column: ## ** This non-numeric index reduces performance **
df_soft_drink = df_soft_drink.set_index('drink')
# print(f'Dataframe with index on one column:\n{df_soft_drink}')
print(f'df_soft_drink.loc[\'water\'] -->\n{df_soft_drink.loc['water']}')
print(f'df_soft_drink.iloc[2] -->\n{df_soft_drink.iloc[2]}')

# ILOC:: Applicable for integer index: Reset index: ## ** This default numeric index enhances performance **
# df_soft_drink = df_soft_drink.reset_index('drink')
# print(f'Dataframe with index reset:\n{df_soft_drink}')

# # Select a single row as a DATAFRAME:
# print(f'df_soft_drink.loc[\'water\'] -->\n{df_soft_drink.loc[['water']]}')
# print(f'df_soft_drink.iloc[i] -->\n{df_soft_drink.iloc[[2]]}')

# # Slicing & Selecting Multiple Rows & Columns:
print(f'df_soft_drink.loc[[\'coffee\', \'smoothie\', \'soda\'], :\'sugar(tsp)\']] -->\n'
      f'{df_soft_drink.loc[['coffee', 'smoothie', 'soda'], :'sugar(tsp)']}')
print(f'df_soft_drink.iloc[2:5, [0, 1, 3]] -->\n{df_soft_drink.iloc[2:5, [0, 1, 3]]}')
