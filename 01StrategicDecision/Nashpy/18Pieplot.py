from matplotlib import pyplot as plt
import numpy as np

plt.style.use('fivethirtyeight')

slices = np.array([45, 15, 20, 20])
labels = ['Hermania\'s\nminimum bargain', '\n\nHermania\'s\ncost of war',
          'Broland\'s\ncost of war', 'Broland\'s\nminimum bargain']
colors = ['#f2c37d', '#cf904a', '#c2bb50', '#f5ee83']
explode = [0, 0.15, 0.15, 0]
labels = [f'{i}\n${j}bn' for i, j in zip(labels, slices)]
plt.pie(slices, labels=labels, colors=colors, explode=explode, shadow=True,
        startangle=90, textprops={'size': 6}, radius=0.6,
        wedgeprops={'edgecolor': 'grey'}, hatch=['', '\\\\', '\\\\', ''])

plt.title("Hermania vs Broland Bargain", fontsize=10)
plt.tight_layout()
# plt.legend(loc='upper left', fontsize=6)
plt.savefig('./plots/20Pieplot.png')
plt.show()

slices = np.array([20, 80])
labels = ['Broland\naccepts Hermania\'s offer', '\nHermania\'s\nremaining share']
colors = ['#f5ee83', '#f2c37d']
explode = [0.15, 0]
labels = [f'{i}\n${j}bn' for i, j in zip(labels, slices)]
plt.pie(slices, labels=labels, colors=colors, explode=explode, shadow=True,
        startangle=90, textprops={'size': 6}, radius=0.6,
        wedgeprops={'edgecolor': 'grey'})

plt.title("Hermania vs BrolandFinal Settlement", fontsize=10)
plt.tight_layout()
# plt.legend(loc='upper left', fontsize=6)
plt.savefig('./plots/20Pieplotsettle.png')
plt.show()
