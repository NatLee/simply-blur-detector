import json

import keras
import numpy as np
import pytest

from blur_detection import DetectionModel, prepare_image
from blur_detection.config import WEB_MODEL_CARD, WEB_MODEL_DIR, WEB_MODEL_FILE
from blur_detection.export import MODEL_CARD_FORMAT, export_web_model, github_provenance, onnx_difference
from blur_detection.images import read_image, write_image

from .helpers import sharp_image

onnxruntime = pytest.importorskip('onnxruntime')


@pytest.fixture
def web_dir(tmp_path):
    return tmp_path / 'web'


@pytest.fixture
def detector():
    keras.utils.set_random_seed(3)
    return DetectionModel()


def test_export_writes_an_onnx_model_that_matches_keras(detector, data_dir, web_dir, monkeypatch):
    monkeypatch.delenv('GITHUB_ACTIONS', raising=False)
    write_image(data_dir / 'demo.png', sharp_image(20))  # the reference picture, like data/demo.png
    card_path = export_web_model(detector.model, str(data_dir), str(web_dir))

    onnx_path = web_dir / WEB_MODEL_DIR / WEB_MODEL_FILE
    assert card_path == web_dir / WEB_MODEL_DIR / WEB_MODEL_CARD
    card = json.loads(card_path.read_text())
    assert card['format'] == MODEL_CARD_FORMAT and card['model'] == WEB_MODEL_FILE
    assert card['classes'] == ['Clear', 'Blur']
    assert card['metrics']['train']['count'] == 6 and card['metrics']['test']['count'] == 4
    for m in card['metrics'].values():
        assert m['accuracy'] == m['correct'] / m['count']  # exact, so a threshold of 0.9 passes at 36 / 40
    # The 300 x 280 test images give 4 x 3 windows of 140 px each (half the shorter side, half overlapping).
    assert card['metrics']['test_windows']['count'] == 4 * 12
    assert card['windows'] == {'fraction': 0.5, 'overlap': 0.5}
    assert card['provenance'] is None
    assert card['versions']['tensorflow'] and card['versions']['onnx']

    # The reference prediction in the card is what the Python pipeline gives for that file ...
    image = prepare_image(read_image(data_dir / 'demo.png'))
    expected = float(np.asarray(detector.predict(image)[0])[0, 1])
    assert card['references'] == {'demo.png': pytest.approx(expected, abs=1e-6)}
    # ... and ONNX Runtime gives the same as Keras.
    session = onnxruntime.InferenceSession(str(onnx_path))
    onnx_p = session.run(None, {session.get_inputs()[0].name: image[np.newaxis]})[0][0, 1]
    assert onnx_p == pytest.approx(expected, abs=1e-5)
    assert onnx_difference(detector.model, onnx_path, image[np.newaxis]) < 1e-5


def test_export_without_a_reference_picture_still_writes_a_card(detector, data_dir, tmp_path):
    card_path = export_web_model(detector.model, str(data_dir), str(tmp_path / 'empty-web'))
    assert json.loads(card_path.read_text())['references'] == {}


def test_provenance_points_at_the_github_actions_run():
    env = {'GITHUB_ACTIONS': 'true', 'GITHUB_SERVER_URL': 'https://github.com', 'GITHUB_REPOSITORY': 'o/r',
           'GITHUB_SHA': 'abc123', 'GITHUB_RUN_ID': '42', 'GITHUB_WORKFLOW': 'Pages'}
    assert github_provenance(env) == {'repository': 'o/r', 'commit': 'abc123', 'run_id': '42',
                                      'run_url': 'https://github.com/o/r/actions/runs/42', 'workflow': 'Pages'}
    assert github_provenance({}) is None
