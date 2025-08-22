from data_handler.malaria_dataset import MalariaDataset
from models.malaria_model import MalariaModel
from utils.visualizer import Visualizer

def main():
    # Load dataset
    dataset = MalariaDataset(im_size=224, batch_size=32)

    # Load pretrained model (instead of training again)
    malaria_model = MalariaModel.load("models/anodiamlenet_continued.keras")

    # Visualize predictions
    visualizer = Visualizer(
        malaria_model,
        dataset.ds,   # the raw TFDS dataset
        dataset.label_names
    )
    visualizer.show_predictions(
        num_samples_to_show=10,
        train_count=int(0.7 * dataset.ds_info.splits["train"].num_examples),
        val_count=int(0.15 * dataset.ds_info.splits["train"].num_examples),
    )

if __name__ == "__main__":
    main()
