import tensorflow as tf
import tensorflow_datasets as tfds

class MalariaDataset:
    def __init__(self, im_size=224, batch_size=32, train_ratio=0.7, val_ratio=0.15):
        self.im_size = im_size
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = 1 - train_ratio - val_ratio

        # Load dataset
        self.ds, self.ds_info = tfds.load(
            'malaria', as_supervised=True, with_info=True, shuffle_files=True
        )
        self.ds = self.ds["train"]
        self.label_names = self.ds_info.features["label"].names

        self._prepare_splits()

    def _resize_rescale(self, image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (self.im_size, self.im_size))
        image = image / 255.0
        return image, label

    def _prepare_splits(self):
        total_examples = self.ds_info.splits["train"].num_examples
        train_count = int(total_examples * self.train_ratio)
        val_count = int(total_examples * self.val_ratio)
        test_count = total_examples - train_count - val_count

        train_ds = self.ds.take(train_count)
        val_ds = self.ds.skip(train_count).take(val_count)
        test_ds = self.ds.skip(train_count + val_count).take(test_count)

        self.train = (
            train_ds.map(self._resize_rescale)
            .shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.val = (
            val_ds.map(self._resize_rescale)
            .shuffle(buffer_size=8, reshuffle_each_iteration=True)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        self.test = (
            test_ds.map(self._resize_rescale)
            .batch(self.batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
