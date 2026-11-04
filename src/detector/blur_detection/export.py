"""
Export a trained model for the web demo: an ONNX file that ONNX Runtime Web runs in the browser, and a
model card (JSON) that says how the model was made and how well it did.
"""
import json
import math
import os
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import keras
import numpy as np
from loguru import logger

from .config import (
    BLUR_CLASS,
    DEMO_IMAGE,
    CLASS_NAMES,
    EXPORT_TOLERANCE,
    PUBLISH_MIN_TEST_ACCURACY,
    PUBLISH_THRESHOLD_NOTE,
    INPUT_SIZE,
    WEB_MODEL_CARD,
    WEB_MODEL_DIR,
    WEB_MODEL_FILE,
    WINDOW_FRACTION,
    WINDOW_OVERLAP,
)
from .images import prepare_image, read_image, sliding_windows
from .utils import find_images, load_images

#: Version of the model card layout; the web demo checks it before reading the card.
MODEL_CARD_FORMAT = 1
#: Packages whose versions go into the model card.
RECORDED_PACKAGES = ('tensorflow', 'keras', 'tf2onnx', 'onnx', 'onnxruntime', 'opencv-python-headless',
                     'blurgenerator')


def export_onnx(model: keras.Model, onnx_path: Path, example: np.ndarray) -> Path:
    """Write ``model`` as an ONNX file with Keras' own exporter (which uses tf2onnx).

    The exporter only takes a model that has run at least once, so ``model`` is first called on
    ``example`` (any batch of model inputs).
    """
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    model(example, training=False)
    model.export(str(onnx_path), format='onnx')
    return onnx_path


def onnx_difference(model: keras.Model, onnx_path: Path, inputs: np.ndarray) -> float:
    """Largest absolute difference between Keras and ONNX Runtime outputs for the same ``inputs``."""
    import onnxruntime  # optional (the export extra); only this check needs it

    session = onnxruntime.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    expected = np.asarray(model(inputs, training=False))
    actual = session.run(None, {session.get_inputs()[0].name: inputs})[0]
    return float(np.abs(actual - expected).max())


def window_inputs(samples) -> Tuple[np.ndarray, np.ndarray]:
    """The heat map windows of every labelled image, prepared, with one-hot labels."""
    inputs, labels = [], []
    for path, label in samples:
        img = read_image(path)
        for x, y, side in sliding_windows(img.shape[1], img.shape[0]):
            inputs.append(prepare_image(img[y:y + side, x:x + side]))
            labels.append(label)
    return np.stack(inputs), keras.utils.to_categorical(labels, len(CLASS_NAMES))


def evaluate(model: keras.Model, data_path: str) -> Dict[str, Dict[str, float]]:
    """Loss and accuracy of ``model`` on every split, and on the heat map windows of the test images."""
    found = find_images(data_path)
    sets = {split: load_images(samples, desc=f'Evaluating {split}...') for split, samples in found.items() if samples}
    if found['test']:
        sets['test_windows'] = window_inputs(found['test'])
    metrics = {}
    for name, (x, y) in sets.items():
        scores = model.evaluate(x, y, verbose=0, return_dict=True)
        # The right answers as a whole number too: accuracy comes back as float32, where 36 / 40 is
        # 0.8999999761…, so a threshold check on it alone can fail at exactly the threshold.
        correct = int(round(float(scores['accuracy']) * len(x)))
        metrics[name] = {'count': len(x), 'correct': correct, 'accuracy': correct / len(x), 'loss': float(scores['loss'])}
    return metrics


#: Committed pictures whose P(blur) the model card records, so that the web demo's tests can check
#: that the page's JavaScript gives the same as Python: the demo picture, drawn by code (data/demo.png).
REFERENCE_IMAGES = (Path(DEMO_IMAGE).name,)


def reference_predictions(model: keras.Model, data_path: str, names: Sequence[str] = REFERENCE_IMAGES) -> Dict[str, float]:
    """P(blur) of each reference picture found in ``data_path``, as the Python pipeline computes it."""
    paths = [Path(data_path) / name for name in names if (Path(data_path) / name).is_file()]
    if not paths:
        return {}
    inputs = np.stack([prepare_image(read_image(path)) for path in paths])
    probabilities = np.asarray(model(inputs, training=False))[:, BLUR_CLASS]
    return {path.name: float(p) for path, p in zip(paths, probabilities)}


def publish_gate(metrics: Dict[str, Dict[str, float]], min_accuracy: float = PUBLISH_MIN_TEST_ACCURACY) -> Dict:
    """Whether the model gets enough of the test pictures right to be published, in whole pictures.

    ``needed`` is the smallest whole number of right answers that reaches ``min_accuracy`` (34 of 40
    at 0.85; the rounding keeps float noise from asking for 35).
    """
    test = metrics.get('test')
    if not test:
        return {'min_test_accuracy': min_accuracy, 'needed': None, 'correct': None, 'passes': False}
    needed = math.ceil(round(min_accuracy * test['count'], 6))
    return {'min_test_accuracy': min_accuracy, 'needed': needed, 'correct': test['correct'],
            'passes': test['correct'] >= needed}


def package_versions(names: Sequence[str] = RECORDED_PACKAGES) -> Dict[str, Optional[str]]:
    """Installed version of each package, or None when it is not installed."""
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def github_provenance(env=os.environ) -> Optional[Dict[str, str]]:
    """Where the model was trained, when this runs in GitHub Actions; None elsewhere."""
    if env.get('GITHUB_ACTIONS') != 'true':
        return None
    server, repo, run_id = env['GITHUB_SERVER_URL'], env['GITHUB_REPOSITORY'], env['GITHUB_RUN_ID']
    return {
        'repository': repo,
        'commit': env['GITHUB_SHA'],
        'run_id': run_id,
        'run_url': f'{server}/{repo}/actions/runs/{run_id}',
        'workflow': env.get('GITHUB_WORKFLOW', ''),
    }


def export_web_model(model: keras.Model, data_path: str, web_dir: str) -> Path:
    """Export ``model`` into the web demo folder and write its model card. Returns the card's path.

    Fails when the ONNX file does not give the same results as Keras (within EXPORT_TOLERANCE).
    """
    web_dir = Path(web_dir)
    probe = np.random.default_rng(0).random((2, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    onnx_path = export_onnx(model, web_dir / WEB_MODEL_DIR / WEB_MODEL_FILE, probe)
    difference = onnx_difference(model, onnx_path, probe)
    if difference > EXPORT_TOLERANCE:
        raise RuntimeError(f'The ONNX model differs from Keras by {difference:.2e} '
                           f'(more than {EXPORT_TOLERANCE:.0e}); not using it.')
    logger.info(f'ONNX and Keras agree within {difference:.2e}.')

    metrics = evaluate(model, data_path)
    card = {
        'format': MODEL_CARD_FORMAT,
        'model': WEB_MODEL_FILE,
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'input': {'size': INPUT_SIZE, 'layout': 'NHWC', 'channels': 'RGB', 'range': [0, 1]},
        'classes': list(CLASS_NAMES),
        'windows': {'fraction': WINDOW_FRACTION, 'overlap': WINDOW_OVERLAP},
        'metrics': metrics,
        'publish': {**publish_gate(metrics), 'note': PUBLISH_THRESHOLD_NOTE},
        'onnx_max_difference': difference,
        'references': reference_predictions(model, data_path),
        'provenance': github_provenance(),
        'versions': package_versions(),
    }
    card_path = onnx_path.with_name(WEB_MODEL_CARD)
    card_path.write_text(json.dumps(card, indent=2) + '\n', encoding='utf-8')
    for split, m in card['metrics'].items():
        logger.info(f'{split}: {m["accuracy"]:.1%} accuracy on {m["count"]} inputs')
    gate = card['publish']
    logger.info(f'Publish gate: {gate["correct"]} of {metrics.get("test", {}).get("count")} test inputs right, '
                f'{gate["needed"]} needed ({gate["min_test_accuracy"]:.0%}): {"passes" if gate["passes"] else "fails"}')
    logger.info(f'Wrote {onnx_path} and {card_path}')
    return card_path

