import keras
import numpy as np
import pytest

from blur_detection import DetectionModel, build_model, get_data
from blur_detection.config import CLASS_NAMES, INPUT_SIZE

#: Weights per layer of the original TensorFlow 2.7 network, worked out from its layer definitions:
#: 3x3 convolutions with 3 -> 128 -> 64 -> 32 channels (+ biases), then 288 -> 16 -> 2 dense layers.
ORIGINAL_PARAMS = [3 * 3 * 3 * 128 + 128, 3 * 3 * 128 * 64 + 64, 3 * 3 * 64 * 32 + 32, 288 * 16 + 16, 16 * 2 + 2]


def random_inputs(n, seed=0):
    return np.random.default_rng(seed).random((n, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)


def test_model_takes_256px_rgb_images_and_gives_two_probabilities():
    model = build_model()
    assert tuple(model.input_shape) == (None, INPUT_SIZE, INPUT_SIZE, 3)
    out = np.asarray(model(random_inputs(3)))
    assert out.shape == (3, len(CLASS_NAMES))
    np.testing.assert_allclose(out.sum(axis=1), 1, rtol=1e-5)


def test_model_has_the_same_layers_as_the_original():
    sizes = [layer.count_params() for layer in build_model().layers if layer.weights]
    assert sizes == ORIGINAL_PARAMS


def test_predict_gives_one_label_per_image():
    # Regression: the label was computed with argmax over the whole batch, so a batch got one label.
    detector = DetectionModel()
    result, text = detector.predict(random_inputs(4))
    assert np.asarray(result).shape == (4, 2)
    assert text == [CLASS_NAMES[i] for i in np.argmax(np.asarray(result), axis=1)]


def test_predict_accepts_a_single_image():
    result, text = DetectionModel().predict(random_inputs(1)[0])
    assert np.asarray(result).shape == (1, 2) and len(text) == 1


@pytest.mark.parametrize('name', ['model.h5', 'model.keras'])
def test_save_then_load_gives_the_same_predictions(tmp_path, name):
    keras.utils.set_random_seed(1)
    detector = DetectionModel()
    path = tmp_path / 'nested' / name
    assert detector.save(str(path)) and path.is_file()
    x = random_inputs(2)
    np.testing.assert_allclose(np.asarray(DetectionModel(str(path)).predict(x)[0]),
                               np.asarray(detector.predict(x)[0]), rtol=1e-6)


def test_a_missing_model_file_starts_an_untrained_model(tmp_path):
    detector = DetectionModel(str(tmp_path / 'nope.h5'))
    assert np.asarray(detector.predict(random_inputs(1))[0]).shape == (1, 2)


def test_test_metrics_are_measured_fresh_every_epoch(data_dir):
    # Regression: the test loss and accuracy were never reset, so every epoch printed a running
    # average over all epochs so far. Now the last epoch's numbers match a fresh evaluation.
    keras.utils.set_random_seed(2)
    detector = DetectionModel()
    train_ds, test_ds = get_data(str(data_dir))
    history = detector.train(train_ds, test_ds, epochs=2)
    fresh = detector.model.evaluate(test_ds, verbose=0, return_dict=True)
    assert history.history['val_loss'][-1] == pytest.approx(fresh['loss'], rel=1e-4)
    assert history.history['val_accuracy'][-1] == pytest.approx(fresh['accuracy'])
