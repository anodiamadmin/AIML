import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Exploratory Data Analysis (EDA)
# # Summarise main characteristics of data & extract important variables
# # Uncover relationships between data & gain better understanding

# Q: What characteristics have the most impact on final price?
# EDA using Descriptive Stats, Grouping, ANOVA, Pearson-Correlation & Correlation Heatmaps

# Descriptive Statistics:
# # df.describe(include='all')        # both numeric and categorical variables
# # categorical_var.value_counts()    # frequency of each value in the categorical var column

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)
df_used_cars = pd.read_csv('./data/01UsedCarsData.csv')
# print(f'Head:\n{df_used_cars.head()}')

# # Frequency Distribution of categorical variables
# print(f'type(df_used_cars) = {type(df_used_cars)}')
df_drive_wheel_counts = df_used_cars[['drive-wheels']]
# print(f'type(df_drive_wheel_counts) = {type(df_drive_wheel_counts)}')
df_drive_wheel_counts = df_drive_wheel_counts.value_counts()
# print(f'df_drive_wheel_counts.value_counts() = {type(df_drive_wheel_counts)}')
df_drive_wheel_counts = df_drive_wheel_counts.to_frame()
# print(f'df_drive_wheel_counts.value_counts().to_frame() = {type(df_drive_wheel_counts)}')
# df_drive_wheel_counts = df_drive_wheel_counts.rename(columns={'drive-wheels': 'value_counts'})
# df_drive_wheel_counts.index.name = 'drive-wheels'
# print(f'df_drive_wheel_counts:\n{df_drive_wheel_counts}')

# # Descriptive Statistics
# print(f'Describe:\n{df_used_cars.describe(include='all')}')
# Box Plots
labels = np.array(['wheel-base', 'length', 'width', 'height'])
df_vehicle_dimensions = df_used_cars[labels]
df_vehicle_dimensions.loc[:, 'length'] *= 100
df_vehicle_dimensions.loc[:, 'width'] *= 100
# print(f'df_vehicle_dimensions\n{df_vehicle_dimensions}')
df_drive_wheel_price = df_used_cars[['drive-wheels', 'price']]
# print(f'df_drive_wheel_price\n{df_drive_wheel_price}')
fig, ax = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(11, 6))
ax[0][0].set_title('Vehicle Dimensions')
box_plt0 = ax[0][0].boxplot(df_vehicle_dimensions, patch_artist=True, showmeans=True, meanline=True,
                      meanprops={'linewidth': 1, 'color': 'blue', 'marker': 'v'},
                      showfliers=True, medianprops={'linewidth': 1, 'color': 'red'},
                      sym='+', flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10}, # Outliers hide=> sym=" "
                      notch=True, widths=0.35, vert=False, labels=labels,
                      boxprops={'linewidth': 1, 'color': 'red', 'facecolor': 'white'},
                      capprops={'linewidth': 1, 'color': 'green'},
                      whiskerprops={'linewidth': 1, 'color': 'purple'})
colors = ['red', 'green', 'blue', 'pink']
for patch, color in zip(box_plt0['boxes'], colors):
    patch.set_facecolor(color)
ax[0][1].set_title('Price Vs Drive Wheel')
colors = ['orange', 'purple', 'yellow']
# Seaborn library is specifically simple over matplotlib for Box-plots
sns.boxplot(x='drive-wheels', y='price', data=df_drive_wheel_price, hue='drive-wheels',
            palette=colors, legend=False, ax=ax[0][1])

# # Scatter plots for relation between 2 continuous variables (engine size = X vs price = Y )
ax[1][0].set_title('Price Vs Engine Size Scatter Plot')
ax[1][0].scatter(df_used_cars[['engine-size']], df_used_cars[['price']])
ax[1][0].set_xlabel('Engine size(cubic-inch)')
ax[1][0].set_ylabel('Price USD($)')

# # Grouping, Pivot Table and Heat Map
# Grouping on single or multiple categorical vars to find
# Relation between the categorical variables vs output the variable
# e.g. Drive wheel system and body style  vs price
# how price varies according to drive-wheel-system and body style
df_style_wheel_price = df_used_cars[['body-style', 'drive-wheels', 'price']]
# print(f'df_style_wheel_price.head():\n{df_style_wheel_price.head()}')
df_grp_mean_style_wheel_price = (df_style_wheel_price
                                 .groupby(['drive-wheels', 'body-style'], as_index=False)
                                 .mean())
df_grp_mean_style_wheel_price = (df_grp_mean_style_wheel_price
                                 .rename(columns={'price': 'group_mean_price'}))
# print(f'df_grp_mean_style_wheel_price:\n{df_grp_mean_style_wheel_price}')
# Pivot Table: But the group by table looks clumsy. So we convert the
# group-by table into a pivot table by the pivot() method
# one variable (drive-wheels) displayed along rows and another variable (body-style) along columns
# The dependent variable is represented in a rectangular grid. Easily visible.
df_pivot = df_grp_mean_style_wheel_price.pivot(index='drive-wheels', columns='body-style')
# print(f'df_pivot:\n{df_pivot}')
# Heat Map: plots target variable (price) over multiple independent variables (drive-wheels, body-style)
ax[1][1].set_title('Heatmap of Price')
heatmap = ax[1][1].pcolor(df_pivot, cmap='RdBu')
plt.colorbar(heatmap, ax=ax[1][1])
ax[1][1].set_xticks(np.arange(df_pivot.shape[1]) + 0.2, minor=False)
ax[1][1].set_yticks(np.arange(df_pivot.shape[0]) + 0.2, minor=False)
ax[1][1].set_xticklabels(sorted(df_used_cars['body-style'].drop_duplicates().values), rotation=30)
ax[1][1].set_yticklabels(sorted(df_used_cars['drive-wheels'].drop_duplicates().values), rotation=30)
ax[1][1].set_xlabel('body-style')
ax[1][1].set_ylabel('drive-wheels')

plt.tight_layout()
plt.savefig('./plots/01EDAStatsGroupPivot.png')
plt.show()
