import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

print("\n✅ Environment check:")
print("TensorFlow version:", tf.__version__)
print("NumPy version:", np.__version__)

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
print("GPUs available:", gpus if gpus else "No GPU detected (using CPU)")

# Quick dataset check (using MNIST since it's small)
ds, info = tfds.load("mnist", as_supervised=True, with_info=True)
print("\n✅ TFDS dataset loaded:", info.name, "| Number of classes:", info.features['label'].num_classes)

# Grab a single example
for image, label in ds['train'].take(1):
    print("\nSample image shape:", image.shape)
    print("Sample label:", label.numpy())

plt.imshow(image.numpy().squeeze(), cmap="gray")
plt.title(f"Sample Label: {label.numpy()}")
plt.show()
