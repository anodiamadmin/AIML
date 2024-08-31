import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

df_used_cars = pd.read_csv('./data/01UsedCarsData.csv')[['symboling', 'normalized-losses',
                                                         'wheel-base', 'length', 'width', 'height',
                                                         'curb-weight', 'engine-size', 'bore',
                                                         'stroke', 'compression-ratio', 'horsepower',
                                                         'peak-rpm', 'city-mpg', 'highway-mpg', 'price',
                                                         'city-L/100km', 'diesel', 'gas']]
df_correlation_matrix = df_used_cars.corr()
print(f'correlation_matrix = \n{df_correlation_matrix}')
sns.heatmap(df_correlation_matrix)

plt.tight_layout()
plt.savefig('./plots/03CorrelationHeatmap.png')
plt.show()
