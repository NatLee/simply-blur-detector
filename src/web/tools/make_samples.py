"""
Draw the sample images of the web demo.

Everything is drawn from code (shapes and OpenCV's built-in Hershey font), so the samples contain no
third-party artwork. The blurred versions are made with blur-generator, the same package that blurs
the training data. Run it in the detector's Docker image from the repository root:

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

WIDTH, HEIGHT = 640, 480
SEED = 7
OUTLINE = (34, 24, 22)  # BGR, nearly black
INDEX_FILE = 'samples.json'

#: Depth values of the depth map: the mascot is near, the hills further away, the city furthest.
NEAR, MIDDLE, FAR = 0, 150, 255


def lerp(a, b, t):
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def draw_scene(rng: np.random.Generator):
    """Return the sharp illustration (BGR) and its depth map (0 = near, 255 = far)."""
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    depth = np.full((HEIGHT, WIDTH), FAR, np.uint8)

    # Dusk sky.
    top, bottom = (84, 36, 40), (96, 150, 246)  # BGR: deep indigo -> warm orange
    for y in range(HEIGHT):
        img[y] = lerp(top, bottom, y / (HEIGHT - 1))

    # Moon and string lights (small bright dots turn into bokeh discs when blurred).
    cv2.circle(img, (520, 92), 46, (196, 236, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (520, 92), 46, OUTLINE, 3, cv2.LINE_AA)
    for x in range(14, WIDTH, 26):
        y = int(40 + 18 * np.sin(x / 70))
        cv2.circle(img, (x, y), 4, (170, 245, 255), -1, cv2.LINE_AA)

    # City skyline with lit windows.
    x = 0
    while x < WIDTH:
        w = int(rng.integers(44, 92))
        h = int(rng.integers(110, 230))
        y0 = 330 - h
        color = lerp((92, 58, 70), (130, 84, 92), rng.random())
        cv2.rectangle(img, (x, y0), (x + w, 330), color, -1)
        cv2.rectangle(img, (x, y0), (x + w, 330), OUTLINE, 2)
        for wy in range(y0 + 10, 320, 16):
            for wx in range(x + 8, x + w - 10, 14):
                if rng.random() < 0.55:
                    cv2.rectangle(img, (wx, wy), (wx + 6, wy + 8), (110, 225, 255), -1)
        x += w + int(rng.integers(2, 10))

    # Hills in the middle distance.
    hills = np.array([[0, 360], [90, 318], [200, 344], [330, 300], [470, 340], [560, 312], [WIDTH, 336],
                      [WIDTH, HEIGHT], [0, HEIGHT]], np.int32)
    cv2.fillPoly(img, [hills], (88, 140, 64), cv2.LINE_AA)
    cv2.polylines(img, [hills[:7]], False, OUTLINE, 3, cv2.LINE_AA)
    cv2.fillPoly(depth, [hills], MIDDLE)
    for _ in range(70):  # grass tufts
        gx, gy = int(rng.integers(0, WIDTH)), int(rng.integers(372, HEIGHT))
        cv2.line(img, (gx, gy), (gx + int(rng.integers(-5, 6)), gy - 9), (52, 104, 40), 2, cv2.LINE_AA)

    draw_mascot(img, depth)

    cv2.putText(img, 'simply-blur-detector', (378, 466), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (240, 244, 248), 2,
                cv2.LINE_AA)
    return img, depth


def draw_mascot(img, depth):
    """A friendly robot in the foreground, with plenty of fine detail to stay sharp."""
    body = np.array([[118, 300], [282, 300], [298, 452], [102, 452]], np.int32)
    head = ((96, 140), (304, 292))
    # The robot is near: mark its outline on the depth map (antenna included).
    cv2.fillPoly(depth, [body], NEAR)
    cv2.rectangle(depth, (head[0][0] - 3, head[0][1] - 3), (head[1][0] + 3, head[1][1] + 3), NEAR, -1)
    cv2.line(depth, (200, 140), (200, 90), NEAR, 22)

    cv2.fillPoly(img, [body], (214, 168, 64), cv2.LINE_AA)
    cv2.polylines(img, [body], True, OUTLINE, 4, cv2.LINE_AA)
    # Checkerboard chest plate.
    for row in range(5):
        for col in range(8):
            if (row + col) % 2 == 0:
                x0, y0 = 136 + col * 16, 330 + row * 12
                cv2.rectangle(img, (x0, y0), (x0 + 15, y0 + 11), (60, 70, 230), -1)
    cv2.rectangle(img, (136, 330), (264, 390), OUTLINE, 2)
    cv2.putText(img, 'SHARP', (142, 430), cv2.FONT_HERSHEY_DUPLEX, 1.15, OUTLINE, 2, cv2.LINE_AA)

    cv2.rectangle(img, *head, (232, 226, 220), -1)
    cv2.rectangle(img, *head, OUTLINE, 4)
    # Antenna.
    cv2.line(img, (200, 140), (200, 104), OUTLINE, 4, cv2.LINE_AA)
    cv2.circle(img, (200, 96), 11, (80, 90, 255), -1, cv2.LINE_AA)
    cv2.circle(img, (200, 96), 11, OUTLINE, 3, cv2.LINE_AA)
    # Eyes with highlights, and a grille mouth.
    for ex in (150, 250):
        cv2.circle(img, (ex, 205), 30, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, (ex, 205), 30, OUTLINE, 4, cv2.LINE_AA)
        cv2.circle(img, (ex + 6, 210), 13, OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(img, (ex + 11, 203), 5, (255, 255, 255), -1, cv2.LINE_AA)
    for mx in range(160, 246, 12):
        cv2.line(img, (mx, 254), (mx, 274), OUTLINE, 3)
    cv2.rectangle(img, (152, 252), (248, 276), OUTLINE, 3)


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
    img, depth = draw_scene(np.random.default_rng(SEED))
    index = []
    for name, title, description, make in SAMPLES:
        cv2.imwrite(str(out / name), make(img, depth))
        index.append({'file': name, 'title': title, 'description': description})
        print('wrote', out / name)
    note = ('Drawn by src/web/tools/make_samples.py (shapes and OpenCV Hershey text, no third-party '
            'artwork); blurred with blur-generator.')
    (out / INDEX_FILE).write_text(json.dumps({'note': note, 'samples': index}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'samples')
