"""The web demo's OpenCV fixture must stay what OpenCV and prepare_image really produce."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

WEB = Path(__file__).resolve().parents[2] / 'web'
FIXTURE = WEB / 'tests' / 'fixtures' / 'opencv-parity.json'
GENERATOR = WEB / 'tools' / 'make_opencv_fixture.py'

pytestmark = pytest.mark.skipif(not FIXTURE.is_file(), reason='the web demo is not next to the detector')


def fresh_fixture():
    spec = importlib.util.spec_from_file_location('make_opencv_fixture', GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_fixture()


def test_fixture_matches_this_opencv():
    committed = json.loads(FIXTURE.read_text(encoding='utf-8'))['cases']
    fresh = fresh_fixture()['cases']
    assert [c['name'] for c in committed] == [c['name'] for c in fresh]
    for old, new in zip(committed, fresh):
        for key in ('rgb', 'equalized', 'gray'):
            assert old[key] == new[key], (old['name'], key)
        for old_r, new_r in zip(old['resized'], new['resized']):
            # OpenCV's SIMD paths may round one level differently on other CPUs.
            assert np.abs(np.subtract(old_r['rgb'], new_r['rgb'])).max() <= 1, old['name']
        assert old['laplacian_variance'] == pytest.approx(new['laplacian_variance'])
        np.testing.assert_allclose(old['model_input']['values'], new['model_input']['values'], atol=1.01 / 255)
