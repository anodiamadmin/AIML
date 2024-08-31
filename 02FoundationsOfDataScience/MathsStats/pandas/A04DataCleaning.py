import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# # Deleting Missing value rows is not a good idea,
# # instead we can impute the missing values with the mean/ median of the column/ group

datafile = './data/A04PreProcessing.csv'

df_real_estate = pd.read_csv(datafile)
# print(df_real_estate['STREET_NUMBER'].isnull())
# print(df_real_estate['BEDROOM_COUNT'].isnull())

# df_real_estate = pd.read_csv(datafile)
# print(df_real_estate['STREET_NUMBER'].isnull()) # Detects standard missing values ['n/a', 'N/A', ''] ONLY
# print(df_real_estate['BEDROOM_COUNT'].isnull())

non_standard_missing_values = ['N/A', '', 'na', 'n/a', 'NaN', '--', 'BMW', '...']
df_real_estate = pd.read_csv(datafile, na_values=non_standard_missing_values)
# print(df_real_estate['GARAGE_COUNT'].isnull())
# print(df_real_estate['BEDROOM_COUNT'].isnull())

# print(df_real_estate['OWNER_OCCUPIED'].isnull())
count = 0
for row in df_real_estate['OWNER_OCCUPIED']:
    try:
        int(row)
        df_real_estate.loc[count, 'OWNER_OCCUPIED'] = np.nan
    except ValueError:
        pass
    count += 1
# print(df_real_estate['OWNER_OCCUPIED'].isnull())

# print(df_real_estate.isnull().sum())
# print(df_real_estate.isnull().values.any())

X_df_feature_matrix = df_real_estate.iloc[:, :-1]
Y_df_target_vector = df_real_estate.iloc[:, -1]
# print(X_df_feature_matrix)
# print(Y_df_target_vector)

# # Impute generates Numpy Matrix (2D Array) from Pandas DataFrame
# impute_BedGarage_Count = SimpleImputer(missing_values=np.nan, strategy='mean')
# impute_BedGarage_Count = impute_BedGarage_Count.fit(df_real_estate.iloc[:, 4:6])
# bedGarage_matrix = impute_BedGarage_Count.transform(df_real_estate.iloc[:, 4:6])
# print(bedGarage_matrix)
#
# impute_Bath_Count = SimpleImputer(missing_values=np.nan, strategy='most_frequent') # mode = 1
# # impute_Bath_Count = SimpleImputer(missing_values=np.nan, strategy='median') # = 1.5
# # impute_Bath_Count = SimpleImputer(missing_values=np.nan, strategy='mean') # mean = 1.75
# impute_Bath_Count = impute_Bath_Count.fit(df_real_estate.iloc[:, 6:7])
# bath_col = impute_Bath_Count.transform(df_real_estate.iloc[:, 6:7])
# print(bath_col)

# fillna() => replaces missing values with a specified value
mean_Bed_count = X_df_feature_matrix['BEDROOM_COUNT'].mean()
X_df_feature_matrix['BEDROOM_COUNT'] = X_df_feature_matrix['BEDROOM_COUNT'].fillna(mean_Bed_count)
median_Garage_count = X_df_feature_matrix['GARAGE_COUNT'].median()
X_df_feature_matrix['GARAGE_COUNT'] = X_df_feature_matrix['GARAGE_COUNT'].fillna(median_Garage_count)
mode_bath_count = X_df_feature_matrix['BATH_COUNT'].mode()[0]
# print(f'mode_bath_count = {mode_bath_count}')
X_df_feature_matrix['BATH_COUNT'] = X_df_feature_matrix['BATH_COUNT'].fillna(mode_bath_count)
X_df_feature_matrix['STREET_NUMBER'] = X_df_feature_matrix['STREET_NUMBER'].fillna(1)
X_df_feature_matrix['OWNER_OCCUPIED'] = X_df_feature_matrix['OWNER_OCCUPIED'].fillna('Y')
print(X_df_feature_matrix)
Y_df_target_vector = (Y_df_target_vector.fillna(Y_df_target_vector.median()))
# print(Y_df_target_vector)