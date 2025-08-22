from data_handler.malaria_dataset import MalariaDataset
from models.malaria_model import MalariaModel
from utils.visualizer import Visualizer

def main():
    # Load dataset
    dataset = MalariaDataset(im_size=224, batch_size=32)

    # Load pretrained model
    malaria_model = MalariaModel.load("models/anodiamlenet_continued.keras")

    # Visualize predictions on processed test data
    visualizer = Visualizer(
        malaria_model,
        dataset.test,   # use processed test dataset
        dataset.label_names
    )
    visualizer.show_predictions(num_samples_to_show=10)

if __name__ == "__main__":
    main()