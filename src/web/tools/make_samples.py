"""
Draw the sample images of the web demo.

The picture is blur_detection.scenes.robot_city_scene, drawn from code (shapes and OpenCV's built-in
Hershey font), so the samples contain no third-party artwork. The blurred versions are made with
blur-generator, the same package that blurs the training data. Run it in the detector's Docker image
from the repository root:

    MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/src/web:/webroot" --entrypoint python \
        simply-blur-detector /webroot/tools/make_samples.py /webroot/public/samples

The output is the same on every run (fixed seed).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from blurgenerator import gaussian_blur, lens_blur_with_depth_map, motion_blur

from blur_detection.scenes import NEAR, SAMPLE_SEED, robot_city_scene

INDEX_FILE = 'samples.json'


#: The samples: file name, title, one-line description, and how to make it from the sharp scene.
SAMPLES = [
    ('portrait.png', 'Portrait mode', 'The robot stays sharp, the city melts into lens bokeh.',
     lambda img, depth: lens_blur_with_depth_map(img, depth, num_layers=6, min_blur=2, max_blur=14,
                                                 focus=NEAR, focus_range=20)),
    ('sharp.png', 'Sharp', 'Crisp edges everywhere: the reference picture.',
     lambda img, depth: img),
    ('shaky.png', 'Shaky hands', 'The whole picture smeared by camera motion.',
     lambda img, depth: motion_blur(img, size=40, angle=18)),
    ('soft.png', 'Soft focus', 'A Gaussian blur over everything.',
     lambda img, depth: gaussian_blur(img, kernel=19, sigma=5)),
]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scene = robot_city_scene(np.random.default_rng(SAMPLE_SEED))
    img, depth = scene.img, scene.depth
    index = []
    for name, title, description, make in SAMPLES:
        cv2.imwrite(str(out / name), make(img, depth))
        index.append({'file': name, 'title': title, 'description': description})
        print('wrote', out / name)
    note = ('Drawn by src/web/tools/make_samples.py with blur_detection.scenes (shapes and OpenCV Hershey '
            'text, no third-party artwork); blurred with blur-generator.')
    (out / INDEX_FILE).write_text(json.dumps({'note': note, 'samples': index}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'samples')
