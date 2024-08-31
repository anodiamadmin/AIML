import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# cmap = ['#5599ff', '#ff9955']
# twodarr = np.linspace(0, 9, 15).reshape(3, 5)
twodarr = np.random.randint(0, 9, 15).reshape(3, 5)
print(twodarr)
annot = np.array([['1a', '1b', '1c', '1d', '1e'], ['2a', '2b', '2c', '2d', '2e'], ['3a', '3b', '3c', '3d', '3e']])

sns.heatmap(twodarr, vmin=0, vmax=10, linewidth=2, linecolor='black', cmap='RdYlGn',
            # annot=True, cmap=cmap,
            annot=annot, fmt='s', annot_kws={'size': 10, 'color': 'red', 'weight': 'bold', 'backgroundcolor': 'yellow'}
            )
# 'viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
# 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
# 'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink', 'spring', 'summer', 'autumn', 'winter', 'cool',
# 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu',
plt.title('HeatMap')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.show()