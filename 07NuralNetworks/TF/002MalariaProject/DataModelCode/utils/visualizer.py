import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import math
from math import floor
import os
from typing import List, Tuple
import tensorflow as tf
from .utils import get_title_from_label

class Visualizer:
    def __init__(self, model, dataset, label_names):
        self.model = model
        self.dataset = dataset
        self.label_names = label_names

    @staticmethod
    def test_infected(self, prediction):
        return "Parasitized" if prediction >= 0.5 else "Uninfected"

    def show_test_sample_by_index(self, test_sample_index, train_examples, val_examples, test_dataset, label_names, BATCH_SIZE):
        # global image_raw, label_raw
        np.set_printoptions(precision=7, suppress=True)

        # Load the original dataset again to access raw images
        ds_malaria_raw, ds_malaria_info_raw = tfds.load('malaria', as_supervised=True, with_info=True, shuffle_files=False)
        ds_malaria_raw = ds_malaria_raw["train"]

        # Calculate the original index of the sample in the raw dataset
        test_start_index = train_examples + val_examples
        original_sample_index = test_start_index + test_sample_index

        # Get the raw image and label for the specified original sample index
        for i, (image_raw, label_raw) in enumerate(ds_malaria_raw):
            if i == original_sample_index:
                break

        batch_num = floor(test_sample_index / BATCH_SIZE)
        sample_in_batch = test_sample_index % BATCH_SIZE

        # Iterate through the processed dataset to get the batch
        for image_batch, label_batch in test_dataset.skip(batch_num).take(1):
            predicted_label = self.test_infected(self.model.predict(np.expand_dims(image_batch[sample_in_batch], axis=0))[0][0])
            print(f"Predicted as: {predicted_label}")

            # label_of_sample = label_batch.numpy()[sample_in_batch]
            # actual_label = label_names[label_of_sample]
            # print(f"Label of the {test_sample_index}-th sample in test_dataset is: {actual_label}")
            print(f"Original index in raw dataset: {original_sample_index}")

            # Display the original raw image
            plt.imshow(image_raw.numpy())
            plt.title(f"Original Image (Actual Label: {label_names[label_raw.numpy()]})")
            plt.axis("off")
            plt.show()

    @staticmethod
    def collect_head_samples(ds: tf.data.Dataset, n: int = 5) -> List[Tuple[np.ndarray, int]]:
        """
        Collect the first n (image, label) pairs from an unbatched TF dataset.
        Returns images as numpy arrays and labels as ints.
        """
        samples: List[Tuple[np.ndarray, int]] = []
        for img, label in ds.take(n):
            img_np = img.numpy()
            lbl = int(label.numpy())
            samples.append((img_np, lbl))
        return samples

    @staticmethod
    def draw_head(
            samples: List[Tuple[np.ndarray, int]],
            save_path: str = "data_viz/sample_training_examples.png",
            figsize: Tuple[int, int] = (5, 5),
    ) -> None:
        """
        Display all samples in rows of up to 5 images each and save the figure.
        """
        if not samples:
            raise ValueError("No samples provided to draw_head().")

        n = len(samples)
        num_cols = 5
        num_rows = (n + num_cols - 1) // num_cols  # ceil(n/5)

        # Scale figure height with number of rows
        fig_w, fig_h = figsize
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_w, fig_h * num_rows))
        axes = np.array(axes).reshape(num_rows, num_cols)
        axes_flat = axes.ravel()

        for idx, ax in enumerate(axes_flat):
            if idx < n:
                img, label = samples[idx]
                # Normalize for display if needed
                img = img.astype(np.float32)
                if img.max() > 1.0:
                    img = img / 255.0

                ax.imshow(img)
                title = get_title_from_label(int(label))
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()

        # Ensure folder exists, then save
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        # Console log success message
        saved_name = os.path.basename(save_path)
        saved_dir = os.path.dirname(os.path.abspath(save_path))
        print(f"Image '{saved_name}' has been saved successfully in '{saved_dir}'.")

        # plt.show()
        # plt.close(fig)

    @staticmethod
    def draw_test_prediction_and_featuremaps(
            model,
            test_ds: tf.data.Dataset,
            index: int,
            im_size: int = 224,
    ) -> None:

        os.makedirs("data_viz", exist_ok=True)

        # Accept trainer wrapper or raw Keras model
        base_model = model.model if hasattr(model, "model") and isinstance(model.model, tf.keras.Model) else model
        if not isinstance(base_model, tf.keras.Model):
            raise TypeError("`model` must be a tf.keras.Model or an object with `.model` (tf.keras.Model).")

        # Fetch the requested sample
        try:
            img, label = next(iter(test_ds.skip(index).take(1)))
        except StopIteration:
            raise IndexError(f"Index {index} is out of range for the provided test dataset.")

        # Preprocess for inference
        img_resized = tf.image.resize(img, [im_size, im_size])
        img_norm = tf.cast(img_resized, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_norm, axis=0)  # (1, H, W, 3)

        # Predict sigmoid score
        prob = float(base_model.predict(img_batch, verbose=0)[0][0])
        pred_label = 1 if prob >= 0.5 else 0
        actual_label = int(label.numpy())
        actual_title = get_title_from_label(actual_label)  # e.g., "parasitized (0)" or "uninfected (1)"
        pred_title = get_title_from_label(pred_label)

        # 1) Save the test sample image with title
        fig = plt.figure(figsize=(5, 5))
        plt.imshow(img.numpy().astype(np.uint8))  # original image from dataset
        plt.axis("off")
        plt.title(f"Actual: {actual_title}; Predicted: {pred_title}; Prediction: {prob:.4f}", fontsize=10)
        plt.tight_layout()
        sample_path = f"data_viz/test_sample_{index}.png"
        plt.savefig(sample_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Collect first two Conv2D layers from the model
        conv_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)]
        if len(conv_layers) < 2:
            raise ValueError("Model must have at least two Conv2D layers to visualize feature maps.")

        # Build a functional graph with new Input and reuse the SAME layer objects to get conv outputs
        inp = tf.keras.Input(shape=(im_size, im_size, 3))
        x = inp
        collected_outputs = []
        conv_seen = 0
        for layer in base_model.layers:
            x = layer(x)
            if isinstance(layer, tf.keras.layers.Conv2D):
                if conv_seen < 2:
                    collected_outputs.append(x)
                conv_seen += 1
            if conv_seen >= 2:
                # We already have outputs for the first two convs; no need to continue
                break

        activation_model = tf.keras.Model(inputs=inp, outputs=collected_outputs[:2])
        act1, act2 = activation_model.predict(img_batch, verbose=0)  # shapes: (1, h1, w1, c1), (1, h2, w2, c2)

        # Helper: robust per-channel normalization to [0,1]
        def _norm01(a: np.ndarray) -> np.ndarray:
            a = a.astype(np.float32)
            vmin, vmax = np.percentile(a, [1, 99])
            if vmax <= vmin:
                return np.zeros_like(a)
            a = np.clip(a, vmin, vmax)
            return (a - vmin) / (vmax - vmin)

        # 2) Save Conv1 feature maps as a 2x3 grid (expecting 6 filters)
        fmap1 = np.squeeze(act1, 0)  # (h1, w1, c1)
        c1 = fmap1.shape[-1]
        rows1, cols1 = 2, 3
        show_c1 = min(c1, rows1 * cols1)
        fig = plt.figure(figsize=(5, 5))
        for i in range(rows1 * cols1):
            ax = plt.subplot(rows1, cols1, i + 1)
            ax.axis("off")
            if i < show_c1:
                ax.imshow(_norm01(fmap1[..., i]), cmap="gray")
                ax.set_title(f"{conv_layers[0].name} ch {i}", fontsize=7)
        plt.tight_layout()
        conv1_path = f"data_viz/featuremaps_conv1_{index}.png"
        plt.savefig(conv1_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 3) Save Conv2 feature maps as a 4x4 grid (expecting 16 filters)
        fmap2 = np.squeeze(act2, 0)  # (h2, w2, c2)
        c2 = fmap2.shape[-1]
        rows2, cols2 = 4, 4
        show_c2 = min(c2, rows2 * cols2)
        fig = plt.figure(figsize=(5, 5))
        for i in range(rows2 * cols2):
            ax = plt.subplot(rows2, cols2, i + 1)
            ax.axis("off")
            if i < show_c2:
                ax.imshow(_norm01(fmap2[..., i]), cmap="gray")
                ax.set_title(f"{conv_layers[1].name} ch {i}", fontsize=7)
        plt.tight_layout()
        conv2_path = f"data_viz/featuremaps_conv2_{index}.png"
        plt.savefig(conv2_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # 4) Console output the sigmoid prediction + save locations
        print(
            f"Prediction (sigmoid) for test index {index}: {prob:.6f}  -> predicted label {pred_label} ({pred_title})")
        print(f"Saved test sample image to:     {os.path.abspath(sample_path)}")
        print(f"Saved Conv1 feature maps (2x3) to: {os.path.abspath(conv1_path)}")
        print(f"Saved Conv2 feature maps (4x4) to: {os.path.abspath(conv2_path)}")
