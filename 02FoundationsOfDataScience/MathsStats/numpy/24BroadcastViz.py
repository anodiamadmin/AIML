import matplotlib.pyplot as plt
import numpy as np
from numpy import array, argmin, sqrt, sum

observation = array([111.0, 188.0])
codes = array([[45.0, 155.0],
               [102.0, 203.0],
               [132.0, 193.0],
               [57.0, 173.0]])
athltDict = {0: 'Female Gymnast',  1: 'Basketball Player', 2: 'Football Lineman', 3: 'Marathon Runner'}
diff = codes - observation    # the broadcast happens here
print(diff)
dist = sqrt(sum(diff**2, axis=-1))
matchMatrix = np.array([observation, codes[argmin(dist)]])
# prediction = athltDict.get(argmin(dist))
# predictionCode = codes[argmin(dist)]
# print(f'Athlete with height, weight = {observation} is closest to {prediction}')
# print(f'{prediction}\'s height is {predictionCode}')
print(f'Match Matrix: {matchMatrix}')

# Graphics Visualization below:
fig, ax = plt.subplots()
fig.set_figheight(6)
fig.set_figwidth(8)

ax.scatter(codes[:, 0], codes[:, 1], label="Athletes", s=50, color="blue", marker="o", linewidth=1, linestyle='-', alpha=0.5)
ax.scatter(observation[0], observation[1], label="Observation", s=50, color="green", marker="o", linewidth=1, linestyle='-', alpha=0.5)
ax.plot(matchMatrix[:, 0], matchMatrix[:, 1], label="Tentative Match", color="red",  marker="*", linewidth=2, linestyle='--', markersize=5)
ax.set_title("Athlete Height Weight Matching")
ax.set_xlabel("Weight (Kg)")
ax.set_ylabel("Height (cm)")
ax.set_xlim(40, 140)
ax.set_ylim(140, 215)
ax.set_xticks(ticks=range(40, 140, 25))
ax.set_yticks(ticks=range(140, 215, 25))
ax.legend(loc="upper right")

plt.grid(True)
plt.tight_layout()
plt.savefig('./plots/24BroadcastViz.png')
plt.show()