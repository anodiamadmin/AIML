import matplotlib.pyplot as plt
import tensorflow as tf

class Visualizer:
    def __init__(self, model, dataset, label_names):
        self.model = model
        self.dataset = dataset
        self.label_names = label_names

    def test_infected(self, prediction):
        return "Parasitized" if prediction >= 0.5 else "Uninfected"

    def show_predictions(self, num_samples_to_show=5):
        print("Predictions on processed test samples:")

        plt.figure(figsize=(15, num_samples_to_show * 3))
        for i, (image, actual_label) in enumerate(self.dataset.unbatch().take(num_samples_to_show)):
            image_batch = tf.expand_dims(image, axis=0)
            prediction = self.model.predict(image_batch, verbose=0)
            predicted_label = self.test_infected(prediction[0][0])

            plt.subplot(num_samples_to_show, 1, i + 1)
            plt.imshow(image.numpy())
            plt.title(f"Actual: {self.label_names[actual_label.numpy()]}, Predicted: {predicted_label}")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
