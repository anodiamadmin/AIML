import numpy as np
import matplotlib.pyplot as plt

sachin_score = np.random.poisson(15, 25)
sourav_score = np.random.randint(10, 50, 25)
# print(f"sourav_score = {sourav_score}\nshape = {sourav_score.shape}")
rahul_score = np.random.poisson(40, 25)
laxman_score = np.random.randint(5, 45, 25)
test_scores = np.array([sachin_score, sourav_score, rahul_score, laxman_score]).transpose()
# print(f"test_scores = {test_scores}\nshape = {test_scores.shape}")
labels = np.array(['Sachin', 'Sourav', 'Rahul', 'Laxman'])
plt.title('Boxplot_values')
box_plt = plt.boxplot(test_scores, patch_artist=True, showmeans=True, meanline=True,
                      meanprops={'linewidth': 1, 'color': 'blue', 'marker': 'v'},
                      showfliers=True, medianprops={'linewidth': 1, 'color': 'red'},
                      sym='+', flierprops={'marker': 'o', 'markerfacecolor': 'red', 'markersize': 10}, # Outliers hide=> sym=" "
                      notch=True, widths=0.35, vert=False, labels=labels,
                      boxprops={'linewidth': 1, 'color': 'red', 'facecolor': 'white'},
                      capprops={'linewidth': 1, 'color': 'green'},
                      whiskerprops={'linewidth': 1, 'color': 'purple'})
colors = ['red', 'green', 'blue', 'pink']
for patch, color in zip(box_plt['boxes'], colors):
    patch.set_facecolor(color)

plt.legend(labels, loc='upper left')

plt.tight_layout()
plt.show()
plt.savefig('./plots/24BoxPlot.png')