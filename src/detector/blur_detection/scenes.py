"""
Pictures drawn from code: the training data in data/.

Everything is made of OpenCV shapes and its built-in Hershey font, so the pictures contain no
third-party artwork. `random_scene` composes a different illustration for every seed (sky, city,
hills, robots, flowers, signs, patterns) on a `Canvas`, which also keeps a depth map (0 = near,
255 = far) for blur-generator's depth-of-field blur. The web demo's sample picture is drawn on a
`Canvas` too, by `sample_scene`.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .config import DATASET_PREFIX, DATASET_SEED, DATASET_SIDE_RANGE, DATASET_SIZES, DEMO_IMAGE, SHARP_FOLDER
from .images import write_image

BGR = Tuple[int, int, int]

OUTLINE: BGR = (34, 24, 22)  # nearly black, like an ink line
WHITE: BGR = (255, 255, 255)
#: Depth map values: robots and flowers are near, hills further away, the sky and the city furthest.
NEAR, MIDDLE, FAR = 0, 150, 255

#: Words that robots wear on their chest and signs show.
WORDS = ('SHARP', 'HELLO', 'FOCUS', 'PIXEL', 'CLEAR', 'LENS', 'ROBO', 'BLUR?', 'EDGE', 'ZOOM', '42')

#: Sky gradients (top, bottom) for random scenes, in BGR.
SKIES: Sequence[Tuple[BGR, BGR]] = (
    ((84, 36, 40), (96, 150, 246)),     # dusk
    ((170, 110, 40), (240, 215, 170)),  # clear day
    ((60, 30, 20), (140, 90, 60)),      # night
    ((150, 190, 120), (215, 240, 250)),  # mint morning
    ((120, 60, 140), (200, 170, 250)),  # pink evening
    ((90, 90, 90), (200, 200, 200)),    # overcast
)
#: Fill colours for buildings, robots, flowers and shapes.
COLORS: Sequence[BGR] = (
    (214, 168, 64), (60, 70, 230), (80, 90, 255), (88, 140, 64), (40, 180, 240), (200, 90, 160),
    (232, 226, 220), (120, 200, 90), (230, 150, 60), (70, 70, 200), (160, 220, 250), (90, 60, 120),
)


def lerp(a: BGR, b: BGR, t: float) -> BGR:
    """The colour a fraction ``t`` of the way from ``a`` to ``b``."""
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


@dataclass
class Canvas:
    """An image to draw on (BGR) and its depth map."""
    width: int
    height: int
    img: np.ndarray = field(init=False)
    depth: np.ndarray = field(init=False)

    def __post_init__(self):
        self.img = np.zeros((self.height, self.width, 3), np.uint8)
        self.depth = np.full((self.height, self.width), FAR, np.uint8)


# ---------------------------------------------------------------- pieces

def sky(c: Canvas, top: BGR, bottom: BGR) -> None:
    """A vertical gradient over the whole picture."""
    for y in range(c.height):
        c.img[y] = lerp(top, bottom, y / (c.height - 1))


def moon(c: Canvas, center: Tuple[int, int], radius: int, color: BGR = (196, 236, 255)) -> None:
    cv2.circle(c.img, center, radius, color, -1, cv2.LINE_AA)
    cv2.circle(c.img, center, radius, OUTLINE, 3, cv2.LINE_AA)


def string_lights(c: Canvas, start: int = 14, step: int = 26, base: int = 40, amplitude: float = 18,
                  period: float = 70, radius: int = 4, color: BGR = (170, 245, 255)) -> None:
    """A wavy row of small bright dots (they bloom into bokeh discs when lens-blurred)."""
    for x in range(start, c.width, step):
        y = int(base + amplitude * np.sin(x / period))
        cv2.circle(c.img, (x, y), radius, color, -1, cv2.LINE_AA)


def skyline(c: Canvas, rng: np.random.Generator, ground: int, widths: Tuple[int, int] = (44, 92),
            heights: Tuple[int, int] = (110, 230), colors: Tuple[BGR, BGR] = ((92, 58, 70), (130, 84, 92)),
            window_color: BGR = (110, 225, 255), window_chance: float = 0.55) -> None:
    """Buildings standing on ``ground`` across the whole width, with lit windows."""
    x = 0
    while x < c.width:
        w = int(rng.integers(*widths))
        h = int(rng.integers(*heights))
        y0 = ground - h
        color = lerp(colors[0], colors[1], rng.random())
        cv2.rectangle(c.img, (x, y0), (x + w, ground), color, -1)
        cv2.rectangle(c.img, (x, y0), (x + w, ground), OUTLINE, 2)
        for wy in range(y0 + 10, ground - 10, 16):
            for wx in range(x + 8, x + w - 10, 14):
                if rng.random() < window_chance:
                    cv2.rectangle(c.img, (wx, wy), (wx + 6, wy + 8), window_color, -1)
        x += w + int(rng.integers(2, 10))


def hills(c: Canvas, ridge: Sequence[Tuple[int, int]], color: BGR = (88, 140, 64)) -> None:
    """Filled hills under the ``ridge`` line (left to right), outlined along the ridge."""
    shape = np.array([*ridge, (c.width, c.height), (0, c.height)], np.int32)
    cv2.fillPoly(c.img, [shape], color, cv2.LINE_AA)
    cv2.polylines(c.img, [shape[:len(ridge)]], False, OUTLINE, 3, cv2.LINE_AA)
    cv2.fillPoly(c.depth, [shape], MIDDLE)


def grass(c: Canvas, rng: np.random.Generator, count: int, top: int, color: BGR = (52, 104, 40)) -> None:
    """Short grass strokes scattered below ``top``."""
    for _ in range(count):
        gx, gy = int(rng.integers(0, c.width)), int(rng.integers(top, c.height))
        cv2.line(c.img, (gx, gy), (gx + int(rng.integers(-5, 6)), gy - 9), color, 2, cv2.LINE_AA)


def robot(c: Canvas, cx: int, top: int, scale: float = 1.0, body: BGR = (214, 168, 64),
          chest: BGR = (60, 70, 230), head: BGR = (232, 226, 220), light: BGR = (80, 90, 255),
          label: str = 'SHARP') -> None:
    """A friendly robot with plenty of fine detail. (cx, top) is the middle of its shoulders."""
    def p(x, y):
        return int(round(cx + scale * x)), int(round(top + scale * y))

    def t(width):
        return max(1, int(round(width * scale)))

    body_shape = np.array([p(-82, 0), p(82, 0), p(98, 152), p(-98, 152)], np.int32)
    head_box = (p(-104, -160), p(104, -8))
    # The robot is near: mark it on the depth map (antenna included).
    cv2.fillPoly(c.depth, [body_shape], NEAR)
    cv2.rectangle(c.depth, p(-107, -163), p(107, -5), NEAR, -1)
    cv2.line(c.depth, p(0, -160), p(0, -210), NEAR, t(22))

    cv2.fillPoly(c.img, [body_shape], body, cv2.LINE_AA)
    cv2.polylines(c.img, [body_shape], True, OUTLINE, t(4), cv2.LINE_AA)
    # Checkerboard chest plate and the label.
    for row in range(5):
        for col in range(8):
            if (row + col) % 2 == 0:
                x0, y0 = -64 + col * 16, 30 + row * 12
                cv2.rectangle(c.img, p(x0, y0), p(x0 + 15, y0 + 11), chest, -1)
    cv2.rectangle(c.img, p(-64, 30), p(64, 90), OUTLINE, t(2))
    cv2.putText(c.img, label, p(-58, 130), cv2.FONT_HERSHEY_DUPLEX, 1.15 * scale, OUTLINE, t(2), cv2.LINE_AA)

    cv2.rectangle(c.img, *head_box, head, -1)
    cv2.rectangle(c.img, *head_box, OUTLINE, t(4))
    # Antenna.
    cv2.line(c.img, p(0, -160), p(0, -196), OUTLINE, t(4), cv2.LINE_AA)
    cv2.circle(c.img, p(0, -204), t(11), light, -1, cv2.LINE_AA)
    cv2.circle(c.img, p(0, -204), t(11), OUTLINE, t(3), cv2.LINE_AA)
    # Eyes with highlights, and a grille mouth.
    for ex in (-50, 50):
        cv2.circle(c.img, p(ex, -95), t(30), WHITE, -1, cv2.LINE_AA)
        cv2.circle(c.img, p(ex, -95), t(30), OUTLINE, t(4), cv2.LINE_AA)
        cv2.circle(c.img, p(ex + 6, -90), t(13), OUTLINE, -1, cv2.LINE_AA)
        cv2.circle(c.img, p(ex + 11, -97), t(5), WHITE, -1, cv2.LINE_AA)
    for mx in range(-40, 46, 12):
        cv2.line(c.img, p(mx, -46), p(mx, -26), OUTLINE, t(3))
    cv2.rectangle(c.img, p(-48, -48), p(48, -24), OUTLINE, t(3))


def flower(c: Canvas, center: Tuple[int, int], radius: int, petals: int, color: BGR, middle: BGR,
           stem: BGR = (52, 120, 40)) -> None:
    """A flower on a stem that reaches the bottom of the picture."""
    x, y = center
    cv2.line(c.img, (x, y), (x, c.height), stem, max(2, radius // 5), cv2.LINE_AA)
    cv2.line(c.depth, (x, y), (x, c.height), NEAR, max(2, radius // 5))
    for k in range(petals):
        a = 2 * np.pi * k / petals
        px, py = int(x + radius * 0.75 * np.cos(a)), int(y + radius * 0.75 * np.sin(a))
        axes = (max(2, radius // 2), max(2, radius // 3))
        angle = float(np.degrees(a))
        cv2.ellipse(c.img, (px, py), axes, angle, 0, 360, color, -1, cv2.LINE_AA)
        cv2.ellipse(c.img, (px, py), axes, angle, 0, 360, OUTLINE, 2, cv2.LINE_AA)
    cv2.circle(c.img, (x, y), max(2, radius // 3), middle, -1, cv2.LINE_AA)
    cv2.circle(c.img, (x, y), max(2, radius // 3), OUTLINE, 2, cv2.LINE_AA)
    cv2.circle(c.depth, (x, y), int(radius * 1.3), NEAR, -1)


def sign(c: Canvas, box: Tuple[int, int, int, int], color: BGR, text: str, text_color: BGR = OUTLINE) -> None:
    """A rectangular sign with a word on it, on a post."""
    x, y, w, h = box
    cv2.line(c.img, (x + w // 2, y + h), (x + w // 2, c.height), OUTLINE, 4)
    cv2.rectangle(c.img, (x, y), (x + w, y + h), color, -1)
    cv2.rectangle(c.img, (x, y), (x + w, y + h), OUTLINE, 3)
    scale = max(0.4, h / 60)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    org = (x + max(4, (w - tw) // 2), y + (h + th) // 2)
    cv2.putText(c.img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, text_color, 2, cv2.LINE_AA)
    cv2.rectangle(c.depth, (x, y), (x + w, c.height), NEAR, -1)


def stars(c: Canvas, rng: np.random.Generator, count: int, bottom: int, color: BGR = (220, 245, 255)) -> None:
    """Small dots and crosses in the sky above ``bottom``."""
    for _ in range(count):
        x, y = int(rng.integers(0, c.width)), int(rng.integers(0, max(1, bottom)))
        if rng.random() < 0.3:
            cv2.line(c.img, (x - 4, y), (x + 4, y), color, 1, cv2.LINE_AA)
            cv2.line(c.img, (x, y - 4), (x, y + 4), color, 1, cv2.LINE_AA)
        else:
            cv2.circle(c.img, (x, y), int(rng.integers(1, 3)), color, -1, cv2.LINE_AA)


def badge(c: Canvas, center: Tuple[int, int], radius: int, sides: int, color: BGR, stripes: BGR) -> None:
    """A striped polygon (a kite, a star of sorts) floating in the scene."""
    x, y = center
    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False) + np.pi / 2
    points = np.array([[int(x + radius * np.cos(a)), int(y - radius * np.sin(a))] for a in angles], np.int32)
    cv2.fillPoly(c.img, [points], color, cv2.LINE_AA)
    mask = np.zeros(c.depth.shape, np.uint8)
    cv2.fillPoly(mask, [points], 255)
    lines = np.zeros_like(c.img)
    for k in range(-radius, radius, 9):
        cv2.line(lines, (x + k - radius, y + radius), (x + k + radius, y - radius), stripes, 3, cv2.LINE_AA)
    inside = (mask > 0) & (lines.any(axis=2))
    c.img[inside] = lines[inside]
    cv2.polylines(c.img, [points], True, OUTLINE, 3, cv2.LINE_AA)


def caption(c: Canvas, text: str, org: Tuple[int, int], scale: float = 0.62, color: BGR = (240, 244, 248)) -> None:
    cv2.putText(c.img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# ---------------------------------------------------------------- whole pictures

def random_scene(rng: np.random.Generator, width: int, height: int) -> Canvas:
    """A random illustration: some of sky, moon, lights, stars, city, hills and grass, then one to
    three things in front (robots, flowers, signs, badges), sometimes a caption."""
    c = Canvas(width, height)
    top, bottom = SKIES[int(rng.integers(len(SKIES)))]
    sky(c, top, bottom)
    ground = int(height * rng.uniform(0.55, 0.75))
    if rng.random() < 0.5:
        stars(c, rng, int(rng.integers(15, 60)), bottom=ground)
    if rng.random() < 0.7:
        r = int(width * rng.uniform(0.05, 0.11))
        moon(c, (int(rng.integers(r, width - r)), int(rng.integers(r, max(r + 1, ground // 2)))), r,
             lerp((196, 236, 255), (120, 210, 250), rng.random()))
    if rng.random() < 0.4:
        string_lights(c, start=int(rng.integers(5, 20)), step=int(rng.integers(18, 32)),
                      base=int(rng.integers(20, 60)), amplitude=float(rng.uniform(5, 25)),
                      period=float(rng.uniform(40, 110)), radius=int(rng.integers(3, 6)))
    if rng.random() < 0.65:
        lo = COLORS[int(rng.integers(len(COLORS)))]
        skyline(c, rng, ground=ground, widths=(width // 14, width // 6), heights=(height // 6, height // 2),
                colors=(lerp(lo, (40, 40, 40), 0.5), lerp(lo, (40, 40, 40), 0.2)),
                window_color=lerp((110, 225, 255), COLORS[int(rng.integers(len(COLORS)))], 0.3),
                window_chance=float(rng.uniform(0.3, 0.8)))
    if rng.random() < 0.85:
        xs = np.linspace(0, width, int(rng.integers(4, 9)))
        ridge = [(int(x), int(ground + rng.integers(-height // 12, height // 12))) for x in xs]
        hills(c, ridge, COLORS[int(rng.integers(len(COLORS)))])
        grass(c, rng, int(rng.integers(20, 90)), top=min(height - 1, ground + height // 15))
    for _ in range(int(rng.integers(1, 4))):
        kind = rng.random()
        if kind < 0.4:
            scale = float(rng.uniform(0.35, 0.8)) * min(width, height) / 480
            robot(c, cx=int(rng.integers(int(100 * scale), width - int(100 * scale))),
                  top=int(height - 150 * scale - rng.integers(0, max(1, int(height * 0.15)))), scale=scale,
                  body=COLORS[int(rng.integers(len(COLORS)))], chest=COLORS[int(rng.integers(len(COLORS)))],
                  head=lerp((232, 226, 220), COLORS[int(rng.integers(len(COLORS)))], 0.25),
                  light=COLORS[int(rng.integers(len(COLORS)))], label=WORDS[int(rng.integers(len(WORDS)))])
        elif kind < 0.65:
            r = int(min(width, height) * rng.uniform(0.05, 0.12))
            flower(c, (int(rng.integers(r, width - r)), int(rng.integers(ground, height - r))), r,
                   int(rng.integers(5, 9)), COLORS[int(rng.integers(len(COLORS)))],
                   COLORS[int(rng.integers(len(COLORS)))])
        elif kind < 0.85:
            w, h = int(width * rng.uniform(0.2, 0.35)), int(height * rng.uniform(0.08, 0.14))
            sign(c, (int(rng.integers(0, width - w)), int(rng.integers(ground - h, height - h - 10)), w, h),
                 COLORS[int(rng.integers(len(COLORS)))], WORDS[int(rng.integers(len(WORDS)))])
        else:
            r = int(min(width, height) * rng.uniform(0.07, 0.14))
            badge(c, (int(rng.integers(r, width - r)), int(rng.integers(r, height - r))), r,
                  int(rng.integers(3, 8)), COLORS[int(rng.integers(len(COLORS)))],
                  COLORS[int(rng.integers(len(COLORS)))])
    if rng.random() < 0.5:
        caption(c, WORDS[int(rng.integers(len(WORDS)))].lower(),
                (int(rng.integers(5, width // 2)), height - int(rng.integers(8, 30))),
                scale=float(rng.uniform(0.5, 1.0)))
    return c


# ---------------------------------------------------------------- the built-in data set

def draw_dataset(data_path: str, sizes: Dict[str, int] = DATASET_SIZES, seed: int = DATASET_SEED) -> List[Path]:
    """Draw the sharp pictures of the built-in data set into ``data_path``.

    Writes ``<split>/nonblur/scene-NNN.png`` for every split in ``sizes`` (numbered across the splits)
    and the demo picture, each a `random_scene` with sides between DATASET_SIDE_RANGE. The same seed
    always gives the same pictures; `generate` then adds the blurred copies. Existing files with the
    same names are replaced, other files are left alone. Returns the written paths.
    """
    rng = np.random.default_rng(seed)
    low, high = DATASET_SIDE_RANGE
    root = Path(data_path)
    targets = [root / split / SHARP_FOLDER for split, count in sizes.items() for _ in range(count)]
    names = [f'{DATASET_PREFIX}-{n:03}.png' for n in range(len(targets))]
    paths = [folder / name for folder, name in zip(targets, names)] + [root / Path(DEMO_IMAGE).name]
    written = []
    for path in paths:
        width, height = (int(v) for v in rng.integers(low, high + 1, 2))
        written.append(write_image(path, random_scene(rng, width, height).img))
    return written
