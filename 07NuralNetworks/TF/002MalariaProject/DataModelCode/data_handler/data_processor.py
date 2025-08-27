import tensorflow_datasets as tfds
from typing import Any   #, List
import tensorflow as tf
# from typing import Tuple
# from matplotlib import pyplot as plt
from typing import Tuple, Union, Dict
# from model_build.malaria_model_trainer import MalariaModelTrainer

class DataProcessor:
    @staticmethod
    def get_malaria_dataset() -> Any:
        ds = tfds.load('malaria', as_supervised=True, with_info=False, shuffle_files=False)
        return ds

    @staticmethod
    def _ensure_dataset(ds_or_splits: Union[tf.data.Dataset, Dict]) -> tf.data.Dataset:
        """Return a concrete tf.data.Dataset from TFDS load output."""
        if isinstance(ds_or_splits, dict):
            # Prefer 'train' if present, otherwise take the first split.
            if 'train' in ds_or_splits:
                return ds_or_splits['train']
            # fallback: first value
            return next(iter(ds_or_splits.values()))
        return ds_or_splits

    @staticmethod
    def count_examples(ds: tf.data.Dataset) -> int:
        """Cardinality with a safe fallback."""
        card = tf.data.experimental.cardinality(ds)
        if card == tf.data.UNKNOWN_CARDINALITY:
            # Rare for TFDS; fallback to counting
            return sum(1 for _ in ds)
        return int(card.numpy())

    @staticmethod
    def split_dataset_test_trainval(
            ds_or_splits: Union[tf.data.Dataset, Dict],
            test_ratio: float = 0.1,
            shuffle_seed: int = 42,
            cache: bool = True,
            prefetch: bool = True
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Split a single-split dataset into (train+val, test) with deterministic shuffling.
        """
        assert 0.0 < test_ratio < 1.0, "test_ratio must be in (0, 1)."
        ds = DataProcessor._ensure_dataset(ds_or_splits)
        total = DataProcessor.count_examples(ds)

        # Deterministic shuffle so the split is random but reproducible
        ds_shuffled = ds.shuffle(buffer_size=total, seed=shuffle_seed, reshuffle_each_iteration=False)

        test_size = int(total * test_ratio)
        test_ds = ds_shuffled.take(test_size)
        trainval_ds = ds_shuffled.skip(test_size)

        if cache:
            test_ds = test_ds.cache()
            trainval_ds = trainval_ds.cache()
        if prefetch:
            autotune = tf.data.AUTOTUNE
            test_ds = test_ds.prefetch(autotune)
            trainval_ds = trainval_ds.prefetch(autotune)

        # print(f"Total: {total} | Test: {test_size} | Train+Val: {total - test_size}")
        return trainval_ds, test_ds

    @staticmethod
    def _preprocess(im: tf.Tensor, label: tf.Tensor, im_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        im = tf.image.resize(im, [im_size, im_size])
        im = tf.cast(im, tf.float32) / 255.0
        label = tf.cast(label, tf.float32)
        return im, label

    @staticmethod
    def to_batches(ds: tf.data.Dataset, im_size: int, batch_size: int,
                   shuffle: bool = False, seed: int = 123) -> tf.data.Dataset:
        if shuffle:
            ds = ds.shuffle(1024, seed=seed, reshuffle_each_iteration=False)
        ds = ds.map(lambda x, y: DataProcessor._preprocess(x, y, im_size),
                    num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    @staticmethod
    def k_fold_split(trainval_ds: tf.data.Dataset, k: int = 5, seed: int = 42):
        """
        Partition a dataset into k folds using index % k. Returns a list of (train_ds, val_ds) pairs.
        """
        # Freeze order deterministically
        card = tf.data.experimental.cardinality(trainval_ds)
        buffer = int(card.numpy()) if card != tf.data.UNKNOWN_CARDINALITY else 10000
        frozen = trainval_ds.shuffle(buffer, seed=seed, reshuffle_each_iteration=False)

        # Attach indices
        enum_ds = frozen.enumerate()

        folds = []
        for fold in range(k):
            # val: i % k == fold
            val_ds = enum_ds.filter(lambda i, _: tf.equal(i % k, fold)).map(lambda i, xy: xy)
            # train: i % k != fold
            train_ds = enum_ds.filter(lambda i, _: tf.not_equal(i % k, fold)).map(lambda i, xy: xy)
            folds.append((train_ds, val_ds))
        return folds
