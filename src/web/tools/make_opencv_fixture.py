"""
Write tests/fixtures/opencv-parity.json: small images and what OpenCV (and the detector's own
prepare_image) make of them. The web demo's tests check that the JavaScript ports in
public/js/imageops.js give the same numbers, and the detector's tests check that this file still
matches OpenCV. Run it in the detector's Docker image from the repository root:

    MSYS_NO_PATHCONV=1 docker run --rm -v "$PWD/src/web:/webroot" --entrypoint python \
        simply-blur-detector /webroot/tools/make_opencv_fixture.py /webroot/tests/fixtures/opencv-parity.json
"""
import json
import sys

import cv2
import numpy as np

from blur_detection.images import prepare_image

#: (name, height, width, seed): random images, odd sizes so that resizing has to interpolate.
IMAGES = [('noise-9x7', 7, 9, 1), ('noise-23x17', 17, 23, 2), ('noise-40x31', 31, 40, 3)]
#: (width, height) targets for cv2.resize: shrink, grow, and both at once.
RESIZE_TARGETS = [(5, 4), (16, 12), (12, 20)]
#: Side of the model input in the fixture (the real model uses 256; small keeps the file small).
MODEL_INPUT_SIZE = 8


def make_image(height, width, seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] // 4 + 100, 0, 255)  # a channel with a narrow histogram
    return img  # RGB


def build_fixture():
    cases = []
    for name, height, width, seed in IMAGES:
        rgb = make_image(height, width, seed)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        cases.append({
            'name': name,
            'width': width,
            'height': height,
            'rgb': rgb.flatten().tolist(),
            'equalized': cv2.merge([cv2.equalizeHist(c) for c in cv2.split(rgb)]).flatten().tolist(),
            'resized': [{'width': w, 'height': h, 'rgb': cv2.resize(rgb, (w, h)).flatten().tolist()}
                        for w, h in RESIZE_TARGETS],
            'gray': gray.flatten().tolist(),
            'laplacian_variance': float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            'model_input': {
                'size': MODEL_INPUT_SIZE,
                'values': prepare_image(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), MODEL_INPUT_SIZE).flatten().tolist(),
            },
        })
    return {'opencv': cv2.__version__, 'cases': cases}


def main(path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(build_fixture(), f, separators=(',', ':'))
        f.write('\n')
    print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1])
