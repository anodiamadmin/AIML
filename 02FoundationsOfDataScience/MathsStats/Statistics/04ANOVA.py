import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ANOVA: Analysis of Variance:
# # Statistical comparison between different groups of categorical variables
# # e.g. Average price of different vehicle makes
# # Anova returns 2 values: F-test score & p-value
# # # F-test score: variation or Distance between sample group means
# # #    F  <  1   =>  Small F-test score  ->  two means of the 2 distributions are not far away
# # #    F  >> 1   =>  Large F-test score  ->  two means of the 2 distributions are far, far away
# # # p-value: Confidence Degree or Overlap between distributions
# # #    p  >  0.5  =>  Large p-value  ->  Much, much overlap between distributions
# # #    p  << 0.5  =>  Small p-value  ->  Not much overlap between distributions

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df_used_cars_make_price = pd.read_csv('./data/01UsedCarsData.csv')[['make', 'price']]
df_make_mean_price = df_used_cars_make_price.groupby('make').mean('price').reset_index()
df_make_mean_price.price = df_make_mean_price.price.round().astype(int)
# print(f'df_make_mean_price = \n{df_make_mean_price}')
df_make_mean_price = df_make_mean_price.sort_values(by=['price'])
# print(f'df_make_mean_price = \n{df_make_mean_price}')
fig, ax = plt.subplots(3, 1, sharex=False, sharey=False, figsize=(11, 6))
ax[0].set_title('Mean Car Price of Brands')
ax[0].bar(df_make_mean_price['make'], df_make_mean_price['price'])
ax[0].set_ylabel('USD')
# ax[0].set_xlabel('Car Makes')
ax[0].set_xticks(df_make_mean_price.make)
ax[0].set_xticklabels(df_make_mean_price.make, rotation=30)

df_honda_price = df_used_cars_make_price.groupby('make').get_group('honda')['price']
# print(f'df_honda_price\n{df_honda_price}')
df_subaru_price = df_used_cars_make_price.groupby('make').get_group('subaru')['price']
# print(f'df_subaru_price\n{df_subaru_price}')
df_jaguar_price = df_used_cars_make_price.groupby('make').get_group('jaguar')['price']
# print(f'df_jaguar_price\n{df_jaguar_price}')
bins = np.arange(0, 50000, 1000)
ax[1].hist(df_honda_price, bins=bins, edgecolor='black', color='red', alpha=.3, linewidth=1)
ax[2].hist(df_subaru_price, bins=bins, edgecolor='black', color='purple', alpha=.3, linewidth=1)
ax[1].hist(df_jaguar_price, bins=bins, edgecolor='black', color='green', alpha=.3, linewidth=1)

f_anova_honda_subaru, p_anova_honda_subaru = stats.f_oneway(df_honda_price, df_subaru_price)
# print(f'anova_honda_subaru: f={round(f_anova_honda_subaru, 6)}, p={round(p_anova_honda_subaru, 6)}')
f_anova_honda_jaguar, p_anova_honda_jaguar = stats.f_oneway(df_honda_price, df_jaguar_price)
# print(f'anova_honda_jaguar: f={round(f_anova_honda_jaguar, 6)}, p={round(p_anova_honda_jaguar, 6)}')
fig.text(.34, .49, bbox=dict(facecolor='yellow', alpha=0.5),
         s=f'Anova: Honda-Jaguar\n'
           f'Distance between means: f={round(f_anova_honda_jaguar, 6)}\n'
           f'Overlap between distributions: p={round(p_anova_honda_jaguar, 6)}')
fig.text(.34, .145, bbox=dict(facecolor='yellow', alpha=0.5),
         s=f'Anova: Honda-Subaru\n'
           f'Distance between means: f={round(f_anova_honda_subaru, 6)}\n'
           f'Overlap between distributions: p={round(p_anova_honda_subaru, 6)}')
fig.text(.16, .5, bbox=dict(facecolor='red', alpha=0.5), s='Honda')
fig.text(.16, .14, bbox=dict(facecolor='purple', alpha=0.5), s='Subaru')
fig.text(.7, .5, bbox=dict(facecolor='green', alpha=0.5), s='Jaguar')

plt.tight_layout()
plt.savefig('./plots/04ANOVA.png')
plt.show()
