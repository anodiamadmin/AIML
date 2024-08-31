import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
plt.style.use('ggplot')

fig1, ax1 = plt.subplots(figsize=(10, 5))
# fig2, ax2 = plt.subplots(figsize=(7, 4))
# fig3, ax3 = plt.subplots(2, 3, sharex=True, sharey=True)
# fig3.set_figheight(6)
# fig3.set_figwidth(12)

userArr = np.array([2, 5, 6, 8, 5, 2, 7, 5, 3, 9, 8])
Xarr = np.arange(len(userArr))
ax1.plot(Xarr, userArr, label="np.array([2,5,6,8,5,2,8])", color="purple",  marker="o", linestyle='-.', linewidth='1', markersize=5)

slice_start = 3
slice_end = 8
dip = 1
sliceX = Xarr[slice_start:slice_end]
sliceArr = userArr[slice_start:slice_end] + dip
ax1.plot(sliceX, sliceArr, label="slice[3:8]", color="red",  marker="o", linestyle='-.', linewidth='1', markersize=5)

slice_start2 = 5
dip2 = 0.5
sliceX2 = Xarr[slice_start2:]
sliceArr2 = userArr[slice_start2:] + dip2
ax1.plot(sliceX2, sliceArr2, label="slice[5:]", color="red",  marker="o", linestyle='-.', linewidth='1', markersize=5)

slice_start3 = -9
slice_end3 = -4
dip3 = -0.5
sliceX3 = Xarr[slice_start3:slice_end3]
sliceArr3 = userArr[slice_start3:slice_end3] + dip3
ax1.plot(sliceX3, sliceArr3, label="slice[-9:-4]", color="green",  marker="o", linestyle='-.', linewidth='1', markersize=5)

slice_start4 = 2
slice_end4 = 9
step4 = 2
dip4 = -1
sliceX4 = Xarr[slice_start4:slice_end4:step4]
sliceArr4 = userArr[slice_start4:slice_end4:step4] + dip4
ax1.plot(sliceX4, sliceArr4, label="slice[2:9:2]", color="orange",  marker="o", linestyle='-.', linewidth='1', markersize=5)

step5 = 3
dip5 = -1.5
sliceX5 = Xarr[::step5]
sliceArr5 = userArr[::step5] + dip5
ax1.plot(sliceX5, sliceArr5, label="slice[::3]", color="blue",  marker="o", linestyle='-.', linewidth='1', markersize=5)

# myList = [1, 2, 3, 4, 5, 6, 7]
# arrFromList = np.array(myList)
# ax1.plot(np.arange(len(arrFromList)), arrFromList, label="np.array([1,2,3,4,5,6,7])", color="blue",  marker="*", linewidth=1, linestyle=':', markersize=5)
#
# arrng = np.arange(8)
# ax1.plot(np.arange(len(arrng)), arrng, label="arange(8)", color="green",  marker="v", linewidth=1, linestyle='--', markersize=5)
#
# arrng_step = np.arange(1, 11, 3)
# ax1.plot(np.arange(len(arrng_step)), arrng_step, label="arange(1,11,3)", color="red",  marker="d", linewidth=1, linestyle='-', markersize=5)
#
# zarr = np.zeros(8)
# ax1.plot(np.arange(len(zarr)), zarr, label="zeros(8)", color="red",  marker="o", linewidth=2, linestyle='-.', markersize=8)
#
# onearr = np.ones(7)
# ax1.plot(np.arange(len(onearr)), onearr, label="ones(7)", color="purple",  marker="*", linewidth=1, linestyle='-', markersize=6)
#
# fullarr = np.full(8, 9)
# ax1.plot(np.arange(len(fullarr)), fullarr, label="full(8,9)", color="green",  marker="d", linewidth=1, linestyle='--', markersize=8)
#
# linspc = np.linspace(1, 10, 7)
# ax1.plot(np.arange(len(linspc)), linspc, label="linspace(1,10,7)", color="orange",  marker="*", linewidth=1, linestyle='-', markersize=7)
#
# geomspc = np.geomspace(1, 10, 7) # Start = 0 gives ERROR
# ax1.plot(np.arange(len(geomspc)), geomspc, label="geomspc(1,10,7)", color="black",  marker="+", linewidth=1, linestyle='--', markersize=7)
#
# # logspc = np.logspace(1, 10, 7)
# # ax2.plot(np.arange(len(logspc)), logspc, label="logspace(1,10,7)", color="red",  marker="d", linewidth=1, linestyle=':', markersize=10)
#
# rndarr = np.random.randint(0,  10, 7)
# ax1.plot(np.arange(len(rndarr)), rndarr, label="rndarr(1,10,7)", color="red",  marker="<", linewidth=1, linestyle='-', markersize=5)

ax1.set_title("1D Array Slicing")
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
# ax2.set_yscale("log")
# ax2.set_xlim(0, 10)
# ax2.set_xticks(ticks=ticks, labels=labels, rotation=45, fontsize=10, ha="right", va="top")
# ax2.legend(loc="upper left")
#
# # # 2D Matrix with linearly increasing values
# diagonal_array = np.diag(range(20))
# zeros_array = np.zeros((15, 12),  dtype=int)
# ones_array = np.ones((15, 12),  dtype=int)
# eye_array = np.eye(20, 15)
# full_array = np.full((20, 55), 10)
# rand_array = np.random.randint(0,  10, (20, 16))
#
# ax3[0][0].matshow(diagonal_array)
# # ax3[0][0] = sns.heatmap(diagonal_array, annot=True, fmt="d", cbar=False)
# ax3[0][0].set_title("2D Diagonal Array")
# ax3[0][0].set_xlabel("Columns")
# ax3[0][0].set_ylabel("Rows")
# ax3[0][0].set_xlim(0, 20)
# ax3[0][0].set_ylim(20, 0)
# ax3[0][0].set_xticks(ticks=range(0, 21, 2))
# ax3[0][0].set_xticks(ticks=range(0, 21, 2))
#
#
# ax3[0][1].matshow(zeros_array)
# ax3[0][1].set_title("2D Zero Array")
#
# ax3[0][2].matshow(ones_array)
# ax3[0][2].set_title("2D Ones Array")
#
# ax3[1][0].matshow(eye_array)
# ax3[1][0].set_title("2D Eye Array")
#
# ax3[1][1].matshow(full_array)
# ax3[1][1].set_title("2D Full Array")
#
# ax3[1][2].matshow(rand_array)
# ax3[1][2].set_title("2D Random Array")

plt.legend(loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()