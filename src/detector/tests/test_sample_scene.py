"""The web demo's sample picture: an AI chip whose traces grow into a neural network."""
import json
from pathlib import Path

import numpy as np
import pytest

from blur_detection.images import read_image
from blur_detection.sample_scene import (
    CHIP_CENTER,
    HEIGHT,
    PIN_REACH,
    SAMPLE_SEED,
    WIDTH,
    ai_chip_scene,
    network_nodes,
    trace_layout,
)
from blur_detection.scenes import FAR, NEAR

from .helpers import laplacian_variance

SAMPLES = Path(__file__).resolve().parents[2] / 'web' / 'public' / 'samples'


@pytest.fixture(scope='module')
def scene():
    return ai_chip_scene(np.random.default_rng(SAMPLE_SEED))


def test_the_scene_is_repeatable(scene):
    again = ai_chip_scene(np.random.default_rng(SAMPLE_SEED))
    np.testing.assert_array_equal(scene.img, again.img)
    np.testing.assert_array_equal(scene.depth, again.depth)
    assert scene.img.shape == (HEIGHT, WIDTH, 3) and scene.depth.shape == (HEIGHT, WIDTH)


def test_the_picture_is_crisp(scene):
    # Far above the usual "blurry" threshold of the Laplacian variance (100), like the training scenes.
    assert laplacian_variance(scene.img) > 300


def test_the_chip_is_near_and_the_network_far(scene):
    cx, cy = CHIP_CENTER
    assert scene.depth[cy, cx] == NEAR                        # the die
    assert scene.depth[cy, cx - PIN_REACH + 2] == NEAR        # a pin on the left
    for layer in network_nodes():
        for x, y in layer:
            assert scene.depth[round(y), round(x)] == FAR
    # Between the chip and the network, the board falls away gradually (a depth of field, not a cut-out).
    row = scene.depth[cy, cx + PIN_REACH:cx + PIN_REACH + 80].astype(int)
    assert np.all(np.diff(row) >= 0) and row[0] < FAR / 4 and row[-1] == FAR


def test_every_trace_bends_and_ends_inside_the_picture():
    # Traces that run straight off the edge looked like motion-blur streaks to the detector, so every
    # one of them must end in a via with room to spare (the right side ends in the network's nodes).
    segments, vias = trace_layout(np.random.default_rng(SAMPLE_SEED))
    margin = 4
    for x, y in vias:
        assert margin <= x <= WIDTH - margin and margin <= y <= HEIGHT - margin, (x, y)
    for (x0, y0), (x1, y1) in segments:
        for x, y in ((x0, y0), (x1, y1)):
            assert 0 <= x <= WIDTH and 0 <= y <= HEIGHT, (x, y)


@pytest.mark.skipif(not SAMPLES.is_dir(), reason='the web demo is not next to the detector')
def test_the_web_samples_come_from_the_chip_scene(scene):
    # tools/make_samples.py draws sharp.png with this module. The picture is rendered with floating
    # point and shrunk by OpenCV, and CPUs round a few values differently (like the blurred copies in
    # data/), so allow one level.
    index = json.loads((SAMPLES / 'samples.json').read_text(encoding='utf-8'))
    assert {s['file'] for s in index['samples']} == {'sharp.png', 'soft.png', 'shaky.png', 'portrait.png'}
    committed = read_image(SAMPLES / 'sharp.png').astype(int)
    assert np.abs(committed - scene.img.astype(int)).max() <= 1
