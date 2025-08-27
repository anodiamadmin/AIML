import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

class MalariaModelTrainer:
    def __init__(self, im_size=224, lr=0.01):
        self.im_size = im_size
        self.lr = lr
        self.model = self._build_model()
        self._compile()

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=(self.im_size, self.im_size, 3)),
            layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(filters=16, kernel_size=5, strides=1, padding='valid', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Flatten(),
            layers.Dense(units=100, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=10, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(units=1, activation='sigmoid')
        ])
        return model

    def _compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.lr),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
        )

    def train(self, train_ds, val_ds, epochs=50, use_early_stopping=False, patience=5,
              min_delta=0.0, monitor="val_loss", mode="min",
              steps_per_epoch=None, validation_steps=None):
        callbacks = []
        if use_early_stopping:
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    patience=patience,
                    min_delta=min_delta,
                    mode=mode,
                    restore_best_weights=True
                )
            )
        return self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
        )

    def evaluate(self, test_ds):
        return self.model.evaluate(test_ds)

    def predict(self, image):
        return self.model.predict(image)

    def save(self, path="model_build/malaria_model.keras"):
        self.model.save(path)

    @staticmethod
    def load(path="model_build/anodiam_le_net_continued.keras"):
        return tf.keras.models.load_model(path)
