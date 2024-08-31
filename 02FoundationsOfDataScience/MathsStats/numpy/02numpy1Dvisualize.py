import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')

fig1, ax1 = plt.subplots(figsize=(10, 5))
# fig2, ax2 = plt.subplots(figsize=(7, 4))

# userArr = np.array([2, 5, 6, 8, 5, 2, 8])
# x_vals = np.arange(len(userArr))
# print(f'userArr = {userArr} :: len(userArr) = {len(userArr)} :: x_vals = {x_vals}')
# ax1.plot(x_vals, userArr, label="np.array([2,5,6,8,5,2,8])", color="purple",
#          marker="o", linestyle='-.', linewidth='1', markersize=5)

# myList = [1, 2, 3, 4, 5, 6, 7]
# arrFromList = np.array(myList)
# ax1.plot(np.arange(len(arrFromList)), arrFromList, label="np.array([1,2,3,4,5,6,7])",
#          color="blue",  marker="*", linewidth=1, linestyle=':', markersize=5)
#
# arrng = np.arange(8)
# ax1.plot(np.arange(len(arrng)), arrng, label="arange(8)", color="green",  marker="v",
#          linewidth=1, linestyle='--', markersize=5)
#
# arrng_step = np.arange(1, 11, 3)
# ax1.plot(np.arange(len(arrng_step)), arrng_step, label="arange(1,11,3)", color="red",
#          marker="d", linewidth=1, linestyle='-', markersize=5)
#
# zarr = np.zeros(8)
# ax1.plot(np.arange(len(zarr)), zarr, label="zeros(8)", color="red",  marker="o",
#          linewidth=2, linestyle='-.', markersize=8)
#
# onearr = np.ones(7)
# ax1.plot(np.arange(len(onearr)), onearr, label="ones(7)", color="purple",  marker="*",
#          linewidth=1, linestyle='-', markersize=6)
#
# fullarr = np.full(8, 9)
# ax1.plot(np.arange(len(fullarr)), fullarr, label="full(8,9)", color="green",  marker="d",
#          linewidth=1, linestyle='--', markersize=8)

# linspc = np.linspace(1, 10, 7)
# print(f'linspc = {linspc}')
# ax1.plot(np.arange(len(linspc)), linspc, label="linspace(1,10,7)", color="orange",  marker="*",
#          linewidth=1, linestyle='-', markersize=7)
#
# geomspc = np.geomspace(1, 10, 7) # Start = 0 gives ERROR
# ax1.plot(np.arange(len(geomspc)), geomspc, label="geomspc(1,10,7)", color="black",  marker="+", linewidth=1, linestyle='--', markersize=7)

# logspc = np.logspace(1, 10, 7)
# print(f'logspc = {logspc}')
# ax2.plot(np.arange(len(logspc)), logspc, label="logspace(1,10,7)", color="red",  marker="d",
#          linewidth=1, linestyle=':', markersize=10)

rndarr = np.random.randint(0,  10, 7)
ax1.plot(np.arange(len(rndarr)), rndarr, label="rndarr(1,10,7)", color="red",  marker="<",
         linewidth=1, linestyle='-', markersize=5)

ax1.set_title("Array Creation Plot")
ax1.set_xlabel("Array Indexes")
ax1.set_ylabel("Array Values")
ax1.set_xlim(0, 10)
ax1.set_ylim(-1, 11)
ticks = np.arange(0, 10, 2)
labels = [f"arr[{tick}]" for tick in ticks]
ax1.set_xticks(ticks=ticks, labels=labels, rotation=45, fontsize=10, ha="right", va="top")
ax1.legend(loc="upper right")

# ax2.set_title("Logspace - Log Scale")
# ax2.set_xlabel("Array Indexes")
# ax2.set_ylabel("Array Values")
# ax2.set_yscale("log")               # Important
# ax2.set_xlim(0, 10)
# ax2.set_xticks(ticks=ticks, labels=labels, rotation=45, fontsize=10, ha="right", va="top")
# ax2.legend(loc="upper left")

plt.grid(True)
plt.tight_layout()
plt.show()