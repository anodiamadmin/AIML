from data_handler.malaria_dataset import MalariaDataset
from models.malaria_model import MalariaModel
from utils.visualizer import Visualizer

def main():
    # Load dataset
    dataset = MalariaDataset(im_size=224, batch_size=32)

    # Initialize a new model (from scratch)
    malaria_model = MalariaModel(im_size=224, lr=0.01)

    # Train the model
    # malaria_model.train(dataset.train, dataset.val, epochs=6)

    # # Evaluate the model
    # malaria_model.evaluate(dataset.test)

    # # Save the trained model
    # malaria_model.save("models/malaria_model.keras")

    # Load the model (if needed)
    malaria_model = MalariaModel.load("models/malaria_model.keras")

    # Visualize predictions
    visualizer = Visualizer(
        malaria_model,
        dataset.test,
        dataset.label_names,
    )
    
    train_count = len(dataset.train)
    val_count = len(dataset.val)
    
    visualizer.show_test_sample_by_index(
        test_sample_index=4001,
        train_examples=train_count,
        val_examples=val_count,
        test_dataset=dataset.test,
        label_names=dataset.label_names,
        BATCH_SIZE=dataset.batch_size
    )

if __name__ == "__main__":
    main()