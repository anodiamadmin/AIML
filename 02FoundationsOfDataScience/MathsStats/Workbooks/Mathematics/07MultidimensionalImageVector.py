import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

img = Image.open("./data/flower.jpg")
img_array = np.asarray(img)

fig = plt.figure()
fig.set_figheight(6)
fig.set_figwidth(6)

original_shape = img_array.shape
vector_image = img_array.flatten()
print(f'img_array.shape = {original_shape} -vs- vector_image.shape = {vector_image.shape}')

ax0 = plt.subplot2grid(shape=(2, 2), loc=(0, 0), colspan=1, rowspan=1)
ax0.imshow(img_array, vmin=0, vmax=255)
ax0.set_title('Flower Image')

# # Vector Arithmatic (Division):
dimmed_vector = (vector_image / 2).astype(int)
dimmed_image = dimmed_vector.reshape(original_shape)
ax1 = plt.subplot2grid(shape=(2, 2), loc=(0, 1), colspan=1, rowspan=1)
ax1.imshow(dimmed_image, vmin=0, vmax=255)
ax1.set_title('Dimmed Image')

btrfly_array = np.asarray(Image.open("./data/butterfly.jpg"))
vector_btrfly = btrfly_array.flatten()
dimmed_btrfly_vector = (vector_btrfly / 2).astype(int)
dimmed_btrfly_image = dimmed_btrfly_vector.reshape(original_shape)
ax2 = plt.subplot2grid(shape=(2, 2), loc=(1, 0), colspan=1, rowspan=1)
ax2.imshow(dimmed_btrfly_image, vmin=0, vmax=255)
ax2.set_title('Dimmed Butterfly')

# # Vector Addition:
superimposed_vector = dimmed_vector + dimmed_btrfly_vector
# Each pixel of the flower image (considered a dimension of the flower image vector)
# is added to the same position pixel of the butterfly image (considered as the
# component of the butterfly image vector in the same dimension).
# The addition of pixel values of the two vectors in the same position has no effect
# on other pixel values.

superimposed_image = superimposed_vector.reshape(original_shape)
ax3 = plt.subplot2grid(shape=(2, 2), loc=(1, 1), colspan=1, rowspan=1)
ax3.imshow(superimposed_image, vmin=0, vmax=255)
ax3.set_title('Superimposed Image')

plt.show()
plt.tight_layout()
plt.savefig(f'./plots/07MultidimensionalImageVector.png')
