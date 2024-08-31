import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from PIL import Image
# pip3 install opencv-python
import cv2

img = Image.open("./data/flower.jpg")
img_array = np.asarray(img)

fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(12)

print(f'img_array: shape = {img_array.shape}')
# print(f'img_array:\n{img_array}')
ax0 = plt.subplot2grid(shape=(3, 4), loc=(0, 0), colspan=1, rowspan=1)
ax0.imshow(img_array, vmin=0, vmax=255)
ax0.set_title('Flower Image')

red_array = img_array[:, :, 0]
print(f'red_array: shape = {red_array.shape}')
# print(f'red_array:\n{red_array}')
red_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('reds', ['#000000', "#ff0000"])
ax1 = plt.subplot2grid(shape=(3, 4), loc=(0, 1), colspan=1, rowspan=1)
ax1.imshow(red_array, cmap=red_cmap, vmin=0, vmax=255)
ax1.set_title('Red')

green_array = img_array[:, :, 1]
print(f'green_array: shape = {green_array.shape}')
# print(f'green_array:\n{green_array}')
green_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('greens', ['#000000', "#00ff00"])
ax2 = plt.subplot2grid(shape=(3, 4), loc=(0, 2), colspan=1, rowspan=1)
ax2.imshow(green_array, cmap=green_cmap, vmin=0, vmax=255)
ax2.set_title('Green')

blue_array = img_array[:, :, 2]
print(f'blue_array: shape = {blue_array.shape}')
# print(f'blue_array:\n{blue_array}')
blue_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blues', ['#000000', "#0000ff"])
ax3 = plt.subplot2grid(shape=(3, 4), loc=(0, 3), colspan=1, rowspan=1)
ax3.imshow(blue_array, cmap=blue_cmap, vmin=0, vmax=255)
ax3.set_title('Blue')

cyan_array = img_array * [0, 1, 1]
print(f'cyan_array: shape = {cyan_array.shape}')
# print(f'cyan_array:\n{cyan_array}')
cyan_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('cyans', ['#000000', "#00ffff"])
ax4 = plt.subplot2grid(shape=(3, 4), loc=(2, 0), colspan=1, rowspan=1)
ax4.imshow(cyan_array, cmap=cyan_cmap, vmin=0, vmax=255)
ax4.set_title('No Red')

yellow_array = img_array * [1, 1, 0]
print(f'yellow_array: shape = {yellow_array.shape}')
# print(f'yellow_array:\n{yellow_array}')
yellow_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('yellows', ['#000000', "#ffff00"])
ax5 = plt.subplot2grid(shape=(3, 4), loc=(2, 1), colspan=1, rowspan=1)
ax5.imshow(yellow_array, cmap=yellow_cmap, vmin=0, vmax=255)
ax5.set_title('No Blue')

gray_img = img.convert("L")
print(f'gray_img:\n{gray_img}')
gray_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('grays', ['#000000', "#ffffff"])
ax6 = plt.subplot2grid(shape=(3, 4), loc=(2, 2), colspan=1, rowspan=1)
ax6.imshow(gray_img, cmap=gray_cmap, vmin=0, vmax=255)
ax6.set_title('Gray')

gray_array = np.asarray(gray_img)
gray_mean = gray_array.mean()
print(f'gray_array: shape = {gray_array.shape} ::: gray_mean = {gray_mean}')
# print(f'gray_array:\n{gray_array}')
duo_array = (gray_array > gray_mean).astype(int) * 255
print(f'duo_array: shape = {duo_array.shape}')
# print(f'duo_array:\n{duo_array}')
duo_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('duo', ['#000000', "#ffffff"])
ax7 = plt.subplot2grid(shape=(3, 4), loc=(2, 3), colspan=1, rowspan=1)
ax7.imshow(duo_array, cmap=duo_cmap, vmin=0, vmax=255)
ax7.set_title('Duo Chrome')

plt.show()
plt.tight_layout()
plt.savefig(f'./plots/33SubplotGrid.png')
