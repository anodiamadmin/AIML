import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
from math import floor

class Visualizer:
    def __init__(self, model, dataset, label_names):
        self.model = model
        self.dataset = dataset
        self.label_names = label_names

    def test_infected(self, prediction):
        return "Parasitized" if prediction >= 0.5 else "Uninfected"

    def show_test_sample_by_index(self, test_sample_index, train_examples, val_examples, test_dataset, label_names, BATCH_SIZE):
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

            label_of_sample = label_batch.numpy()[sample_in_batch]
            actual_label = label_names[label_of_sample]
            # print(f"Label of the {test_sample_index}-th sample in test_dataset is: {actual_label}")
            print(f"Original index in raw dataset: {original_sample_index}")

            # Display the original raw image
            plt.imshow(image_raw.numpy())
            plt.title(f"Original Image (Actual Label: {label_names[label_raw.numpy()]})")
            plt.axis("off")
            plt.show()

