"""
Blur detection model: a small convolutional network that tells sharp images from blurry ones.
"""
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import keras
import numpy as np
from keras import layers
from loguru import logger

from .config import CLASS_NAMES, INPUT_SIZE

#: Shape of a batch of model inputs: any number of RGB images of INPUT_SIZE x INPUT_SIZE.
INPUT_SHAPE = (None, INPUT_SIZE, INPUT_SIZE, 3)


def build_model(input_shape: Sequence[Optional[int]] = INPUT_SHAPE) -> keras.Model:
    """Build the (untrained) network.

    Three convolution + max-pooling blocks shrink a 256 x 256 image to 3 x 3 x 32 features, then two
    dense layers turn them into the probabilities of ``CLASS_NAMES``. The layers are the same as in the
    original TensorFlow 2.7 version of this project.
    """
    return keras.Sequential([
        keras.Input(shape=tuple(input_shape[1:]), name='image'),
        layers.Conv2D(128, 3, activation='selu', name='conv1'),
        layers.MaxPool2D(2, name='pool1'),
        layers.Conv2D(64, 3, activation='selu', name='conv2'),
        layers.MaxPool2D(4, name='pool2'),
        layers.Conv2D(32, 3, activation='selu', name='conv3'),
        layers.MaxPool2D(8, name='pool3'),
        layers.Flatten(name='flatten'),
        layers.Dense(16, activation='selu', name='dense1'),
        layers.Dense(len(CLASS_NAMES), activation='softmax', name='probabilities'),
    ], name='blur_detector')


class EpochLogger(keras.callbacks.Callback):
    """Log the loss and accuracy of every epoch, for the training and the test data."""

    def __init__(self, epochs: int):
        super().__init__()
        self.epochs = epochs

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(f'Epoch {epoch + 1} / {self.epochs}')
        logger.info('Loss: {:.5f}, Acc: {:.2f} % | Test Loss: {:.5f}, Test Acc: {:.2f} %'.format(
            logs.get('loss', float('nan')), logs.get('accuracy', float('nan')) * 100,
            logs.get('val_loss', float('nan')), logs.get('val_accuracy', float('nan')) * 100))


class DetectionModel():
    """
    Detection module
    ...

    Attributes
    ----------
    @ model : keras.Model
        The network (see `build_model`).

    Methods
    -------
    @ train(train_ds, test_ds, epochs=10)
        Train with blur and non blur data.
    @ predict(imgs)
        Predict images.
    @ save(model_dir)
        Save the model to a ``.h5`` or ``.keras`` file.
    """
    def __init__(self, model_dir: Optional[str] = None, input_shape: Sequence[Optional[int]] = INPUT_SHAPE):
        """Build the network and, when ``model_dir`` is an existing file, load its weights.

        ``model_dir`` can be a file written by `save` (``.h5`` or ``.keras``) or the ``.h5`` weights
        file of the original TensorFlow 2.7 version. Leave it out (or point it at a missing file) to
        start from an untrained network.
        """
        self.model = build_model(input_shape)
        self.model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=[keras.metrics.CategoricalAccuracy(name='accuracy')],
        )
        if model_dir and Path(model_dir).is_file():
            self.model.load_weights(model_dir)
            logger.info('Blur Detection Model Loaded.')
        else:
            logger.warning('Cannot find blur detection model. Please check model directory.')

    def save(self, model_dir: str) -> bool:
        """Save the model. The extension picks the format: ``.keras`` (Keras) or ``.h5`` (HDF5)."""
        Path(model_dir).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(model_dir)
        return True

    def train(self, train_ds, test_ds, epochs: int = 10) -> keras.callbacks.History:
        """Train on ``train_ds`` and check against ``test_ds`` after every epoch (see `get_data`)."""
        # The datasets shuffle themselves (see get_data), so Keras does not need to.
        return self.model.fit(train_ds, validation_data=test_ds, epochs=epochs, shuffle=False, verbose=0,
                              callbacks=[EpochLogger(epochs)])

    def predict(self, imgs) -> Tuple[Any, List[str]]:
        """Predict one prepared image (H x W x 3) or a batch of them (N x H x W x 3).

        Returns the class probabilities as a TensorFlow tensor (N x 2, columns in ``CLASS_NAMES``
        order; ``np.asarray`` turns it into an array) and one label per image, ``'Clear'`` or ``'Blur'``.
        """
        imgs = np.asarray(imgs, dtype=np.float32)
        if imgs.ndim == 3:
            imgs = imgs[np.newaxis]
        result = self.model(imgs, training=False)
        text = [CLASS_NAMES[index] for index in np.argmax(result, axis=-1)]
        return result, text
