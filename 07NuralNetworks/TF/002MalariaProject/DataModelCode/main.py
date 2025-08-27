import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import math
from data_handler.data_processor import DataProcessor
from model_build.malaria_model_trainer import MalariaModelTrainer
from utils.visualizer import Visualizer
import tensorflow as tf
import shutil

def main():
    # get data set
    ds = DataProcessor.get_malaria_dataset()
    trainval_ds, test_ds = DataProcessor.split_dataset_test_trainval(ds, test_ratio=0.1)

    # ---- Visualize a few training samples (up to 5 per row, figsize=(7,7)) ----
    samples = Visualizer.collect_head_samples(trainval_ds, n=20)
    Visualizer.draw_head(samples)  # saves to data_viz/sample_training_examples.png

    # Hyperparams
    k = 5
    im_size = 224
    batch_size = 32
    lr = 0.01
    epochs = 50

    # Prepare test pipeline once (finite; do NOT repeat)
    test_ds_prepared = DataProcessor.to_batches(test_ds, im_size, batch_size, shuffle=False)

    # # Build the folds for training
    # folds = DataProcessor.k_fold_split(trainval_ds, k=k, seed=42)
    #
    # # Total examples in train+val (for exact fold sizes and steps computation)
    # total = DataProcessor.count_examples(trainval_ds)
    #
    # fold_histories = []
    # fold_test_metrics = []
    #
    # # Track best-by-test-accuracy
    # best_model = None
    best_fold = None
    # best_test_acc = -1.0
    # per_fold_scores = []  # [(fold_idx, test_loss, test_acc)]
    #
    # for fold_idx, (train_raw, val_raw) in enumerate(folds, start=1):
    #     print(f"\n===== Fold {fold_idx}/{k} =====")
    #
    #     # Pipelines: make train/val datasets infinite; we'll bound epoch length via steps_*
    #     train_ds = DataProcessor.to_batches(
    #         train_raw, im_size, batch_size, shuffle=True, seed=42 + fold_idx
    #     ).repeat()
    #     val_ds = DataProcessor.to_batches(
    #         val_raw, im_size, batch_size, shuffle=False
    #     ).repeat()
    #
    #     # Compute exact fold sizes (enum%k split -> first r folds have one extra example)
    #     q, r = divmod(total, k)
    #     val_size = q + (1 if (fold_idx - 1) < r else 0)
    #     train_size = total - val_size
    #
    #     steps_per_epoch = math.ceil(train_size / batch_size)
    #     validation_steps = math.ceil(val_size / batch_size)
    #
    #     # Fresh model per fold
    #     malaria_model = MalariaModelTrainer(im_size=im_size, lr=lr)
    #
    #     # Train with early stopping and explicit steps to avoid "input ran out of data" warnings
    #     history = malaria_model.train(
    #         train_ds,
    #         val_ds,
    #         epochs=epochs,
    #         use_early_stopping=True,
    #         patience=5,
    #         min_delta=0.0,
    #         monitor="val_loss",
    #         mode="min",
    #         steps_per_epoch=steps_per_epoch,
    #         validation_steps=validation_steps,
    #     )
    #     fold_histories.append(history.history)
    #
    #     # Evaluate on the held-out test split (constant across folds; finite dataset)
    #     test_metrics = malaria_model.evaluate(test_ds_prepared)
    #     fold_test_metrics.append(test_metrics)
    #
    #     # Unpack test loss/accuracy robustly
    #     if isinstance(test_metrics, dict):
    #         test_loss = float(test_metrics.get("loss", float("nan")))
    #         test_acc = float(
    #             test_metrics.get("accuracy", test_metrics.get("binary_accuracy", float("nan")))
    #         )
    #     elif isinstance(test_metrics, (list, tuple)):
    #         # Keras typically returns [loss, accuracy] when compiled with metrics=[BinaryAccuracy(...)]
    #         test_loss = float(test_metrics[0]) if len(test_metrics) >= 1 else float("nan")
    #         test_acc = float(test_metrics[1]) if len(test_metrics) >= 2 else float("nan")
    #     else:
    #         # Fallback: if a single scalar is returned, treat it as accuracy
    #         test_loss = float("nan")
    #         test_acc = float(test_metrics)
    #
    #     per_fold_scores.append((fold_idx, test_loss, test_acc))
    #     print(f"Fold {fold_idx} → test_loss={test_loss:.4f}, test_accuracy={test_acc:.4f}")
    #
    #     # Keep the best model by test accuracy
    #     if test_acc > best_test_acc:
    #         best_test_acc = test_acc
    #         best_fold = fold_idx
    #         best_model = malaria_model  # keep the trainer; we'll call .save or .model.save later
    #
    # # Summary across folds
    # print("\n=== Cross-Validation Test Accuracies ===")
    # for f, tl, ta in per_fold_scores:
    #     print(f"Fold {f}: test_accuracy={ta:.4f} (test_loss={tl:.4f})")
    #
    # # Save only the best-by-test-accuracy model
    # if best_model is None:
    #     raise RuntimeError("No folds were trained; cannot save model.")
    #
    # os.makedirs("saved_models", exist_ok=True)
    # model_path = f"saved_models/malaria_model.keras"
    # if hasattr(best_model, "save"):
    #     best_model.save(model_path)
    # else:
    #     best_model.model.save(model_path)
    #
    # print(
    #     f"\nSaved best model from fold {best_fold} to: {os.path.abspath(model_path)} "
    #     f"because it achieved the highest test accuracy = {best_test_acc:.4f} among all folds."
    # )

    # loading a saved model
    # Paths
    model_path = f"saved_models/malaria_model_best_fold{best_fold}.keras"
    canonical_path = "saved_models/malaria_model.keras"

    # Ensure a canonical copy exists; re-save from the best-fold file if needed
    if not os.path.exists(canonical_path):
        tmp = tf.keras.models.load_model(model_path)
        tmp.save(canonical_path)

    # Load canonical model
    loaded_best_model = tf.keras.models.load_model(canonical_path)
    print(f"Reloaded model from: {os.path.abspath(canonical_path)}")

    # Evaluate on the held-out test set
    metrics = loaded_best_model.evaluate(test_ds_prepared, return_dict=True, verbose=0)
    test_acc_loaded = float(metrics.get("accuracy", metrics.get("binary_accuracy", float('nan'))))
    print(f"[Reloaded] Test accuracy on held-out test_ds: {test_acc_loaded:.4f}")

    Visualizer.draw_test_prediction_and_featuremaps(
        model=loaded_best_model,
        test_ds=test_ds,  # pass the raw/unbatched test dataset, not the batched pipeline
        index=16,  # <- choose any valid index
        im_size=224
    )

if __name__ == "__main__":
    main()
