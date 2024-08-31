import numpy as np
from matplotlib import pyplot as plt

plt.style.use('fivethirtyeight')

days = np.arange(1, 21)
handingOver = [8, 6, 5, 5, 6, 5, 4, 5, 6, 4, 4, 2, 3, 3, 2, 1, 1, 1, 2, 0]
handedOver1 = [0, 1, 2, 1, 1, 2, 2, 2, 1, 2, 2, 4, 3, 3, 3, 4, 4, 4, 3, 4]
handedOver2 = [0, 1, 1, 2, 1, 1, 2, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4]

labels = ['Anirban', 'Sayan', 'Debasish']

plt.stackplot(days, handingOver, handedOver1, handedOver2,
              labels=labels, colors=['#ff7f0e', '#1f77b4', '#2ca02c'])

plt.title("Handover Hours of AI Project")
plt.xlabel("Days")
plt.ylabel("Hours")
x_ticks = np.arange(1, len(days), 5)
# print(f"x_ticks = ", x_ticks)
plt.xticks(ticks=x_ticks,  rotation=-45)
plt.tight_layout()
plt.legend(loc=(0.1, 0.05))
plt.savefig('./plots/06StackplotFixedtotal.png')
plt.show()