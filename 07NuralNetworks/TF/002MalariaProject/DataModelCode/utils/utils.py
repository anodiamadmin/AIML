# utils/utils.py
from typing import Tuple, Union
import time
import tensorflow as tf

def get_title_from_label(label: int) -> str:
    """
    Return a display title for the given label.
    Logic: "parasitized (0)" if label == 0 else "uninfected (1)".
    """
    return "parasitized (0)" if label == 0 else "uninfected (1)"


def _ensure_keras_model(model) -> tf.keras.Model:
    """Allow passing either a tf.keras.Model or a trainer with `.model`."""
    if hasattr(model, "model") and isinstance(model.model, tf.keras.Model):
        return model.model
    if isinstance(model, tf.keras.Model):
        return model
    raise TypeError("`model` must be a tf.keras.Model or an object exposing `.model` of type tf.keras.Model.")


def _prep_single_sample(sample: Union[tuple, list, tf.Tensor], im_size: int) -> tf.Tensor:
    """Accepts (image, label) or image tensor; returns a (1,H,W,3) batch, resized and normalized."""
    img = sample[0] if isinstance(sample, (tuple, list)) else sample
    img_resized = tf.image.resize(img, [im_size, im_size])
    img_norm = tf.cast(img_resized, tf.float32) / 255.0
    return tf.expand_dims(img_norm, axis=0)  # (1, H, W, 3)


def prediction_time(model, sample: Union[tuple, list, tf.Tensor], im_size: int = 224) -> Tuple[int, float, float]:
    """
    Run inference on a single test sample and return:
        (pred_label, prob, elapsed_ms)

    pred_label: 0 or 1 (threshold=0.5)
    prob: sigmoid output in [0,1]
    elapsed_ms: time to run the prediction (milliseconds)
    """
    m = _ensure_keras_model(model)
    x = _prep_single_sample(sample, im_size)

    # Optional warm-up to avoid first-call graph/setup overhead
    _ = m.predict(x, verbose=0)

    start = time.perf_counter()
    y = m.predict(x, verbose=0)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    prob = float(y[0][0])
    pred_label = 1 if prob >= 0.5 else 0
    return pred_label, prob, elapsed_ms
