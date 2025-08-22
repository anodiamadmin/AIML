import matplotlib.pyplot as plt
import tensorflow as tf

class Visualizer:
    def __init__(self, model, dataset, label_names):
        self.model = model
        self.dataset = dataset
        self.label_names = label_names

    def test_infected(self, prediction):
        return "Parasitized" if prediction >= 0.5 else "Uninfected"

    def show_predictions(self, num_samples_to_show=5, train_count=0, val_count=0):
        print("Predictions on individual raw and processed test samples side-by-side:")

        plt.figure(figsize=(15, num_samples_to_show * 3))  # Dynamic figure size
        test_start_index = train_count + val_count

        for i, (orig_image, actual_label) in enumerate(
            self.dataset.skip(test_start_index).take(num_samples_to_show)
        ):
            # Preprocess image for prediction
            processed_image = tf.image.resize(tf.cast(orig_image, tf.float32), (224, 224)) / 255.0
            processed_image_batch = tf.expand_dims(processed_image, axis=0)

            prediction = self.model.predict(processed_image_batch, verbose=0)
            predicted_label = self.test_infected(prediction[0][0])

            # Original image
            plt.subplot(num_samples_to_show, 2, 2 * i + 1)
            plt.imshow(orig_image.numpy())
            plt.title(f"Actual: {self.label_names[actual_label.numpy()]}")
            plt.axis("off")

            # Processed image
            plt.subplot(num_samples_to_show, 2, 2 * i + 2)
            plt.imshow(processed_image.numpy())
            plt.title(f"Predicted: {predicted_label}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
