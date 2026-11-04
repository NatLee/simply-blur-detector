"""
The web demo's sample picture: an AI chip whose circuit traces grow into a neural network.

It is drawn like a vector illustration: dark ink outlines, flat colours and crisp cel shading on a
dusk-gradient sky with a blueprint grid. That style is what the detector learns "sharp" from (the
training scenes in `scenes` are ink-outlined cartoons too). An earlier neon version with soft glows
on a near-black sky looked blurry to the model, window by window, even though nothing in it was
blurred.

Shapes are signed distance fields and OpenCV lines rendered at SUPERSAMPLING times the final size and
shrunk with INTER_AREA, which gives smooth anti-aliased edges. `ai_chip_scene` returns a `Canvas`
(picture and depth map) like the other scenes: the chip is near, the board falls away from it, the
network is far, so blur-generator's depth-of-field blur keeps the chip sharp and melts the rest.

Everything is drawn from code: no photos, logos or third-party artwork.
"""
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .scenes import FAR, NEAR, Canvas

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

#: Seed of the sample picture (sparkles, trace lengths, lit connections, weights, the code on the chip).
SAMPLE_SEED = 7
#: Size of the picture, in pixels. The layout below is drawn for this size.
WIDTH, HEIGHT = 640, 480
#: Drawing happens at this multiple of the final size; shrinking it back smooths every edge.
SUPERSAMPLING = 2


def _rgb(hex_color: str) -> np.ndarray:
    return np.array([int(hex_color[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255


#: Colours (RGB). The sky is a list of (position from top to bottom, colour) stops.
SKY = [(0.0, _rgb('#1b2160')), (0.55, _rgb('#5a3ea6')), (1.0, _rgb('#e98fa8'))]
INK = _rgb('#14121c')
WHITE = _rgb('#ffffff')
PACKAGE, PACKAGE_LIT, PACKAGE_SHADE, PACKAGE_TEXT = _rgb('#363c58'), _rgb('#4d5680'), _rgb('#262a3e'), _rgb('#c9d0ec')
GOLD, GOLD_LIT = _rgb('#f0b545'), _rgb('#ffe29a')
DIE, DIE_CORE, DIE_RIM = _rgb('#0e2f52'), _rgb('#2fd6e6'), _rgb('#5ff0ff')
TRACE, LETTER_SHADOW = _rgb('#8ef5d0'), _rgb('#ff7ad9')
NODE_COLORS = (_rgb('#7ff3ff'), _rgb('#ff9ae4'))
PLATE = _rgb('#f5f2ff')

# Layout, in pixels of the final picture.
CHIP_CENTER = (196, 244)
PACKAGE_HALF, PACKAGE_RADIUS, BEVEL = 104, 14, 8
PIN_COUNT, PIN_PITCH, PIN_WIDTH, PIN_LENGTH = 10, 16, 8, 18
#: How far the pin tips reach from the chip's centre.
PIN_REACH = PACKAGE_HALF + PIN_LENGTH - 2
DIE_HALF, CORE_CELL = 60, 10
#: Traces leave each pin straight, bend 45 degrees and end in a via, all before the picture's edge:
#: traces that run off the edge read as motion-blur streaks to the detector.
TRACE_FIRST_STRAIGHT, TRACE_BEND, TRACE_END, VIA_CLEARANCE = 6, 16, (6, 18), 12
#: The network: x of every layer and its number of nodes; the last layer is the network's answer.
LAYERS = [(392, 5), (464, 7), (536, 6), (604, 2)]
NETWORK_TOP, NETWORK_BOTTOM = 70, 400
OUTPUTS = ('CLEAR 0.98', 'BLUR 0.02')
GRID_STEP = 32
SPARKLES = 26
CAPTION = ('simply-blur-detector', (432, 462))
#: Depth grows by this much per pixel of distance from the chip, up to FAR.
DEPTH_PER_PIXEL = 4.0


class Painter:
    """A float RGB picture at SUPERSAMPLING times the final size. Coordinates are final pixels."""

    def __init__(self, width: int, height: int):
        self.w, self.h = width * SUPERSAMPLING, height * SUPERSAMPLING
        self.img = np.zeros((self.h, self.w, 3), np.float32)
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.x, self.y = (xs + 0.5) / SUPERSAMPLING, (ys + 0.5) / SUPERSAMPLING

    def coverage(self, sdf: np.ndarray) -> np.ndarray:
        """How much of each pixel a shape covers, from its signed distance (negative inside)."""
        return np.clip(0.5 - sdf * SUPERSAMPLING, 0, 1)

    def lines(self, segments: Sequence[Segment], width: float) -> np.ndarray:
        """Coverage of anti-aliased line segments ``width`` pixels wide."""
        mask = np.zeros((self.h, self.w), np.uint8)
        k = SUPERSAMPLING * 16  # OpenCV's fixed-point coordinates (shift=4)
        for (x0, y0), (x1, y1) in segments:
            cv2.line(mask, (round(x0 * k), round(y0 * k)), (round(x1 * k), round(y1 * k)), 255,
                     max(1, round(width * SUPERSAMPLING)), cv2.LINE_AA, shift=4)
        return mask.astype(np.float32) / 255

    def text(self, text: str, origin: Point, scale: float, thickness: float = 1.0) -> np.ndarray:
        """Coverage of a line of OpenCV Hershey text; ``origin`` is its bottom left corner."""
        mask = np.zeros((self.h, self.w), np.uint8)
        cv2.putText(mask, text, (round(origin[0] * SUPERSAMPLING), round(origin[1] * SUPERSAMPLING)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale * SUPERSAMPLING, 255, max(1, round(thickness * SUPERSAMPLING)),
                    cv2.LINE_AA)
        return mask.astype(np.float32) / 255

    def paint(self, mask: np.ndarray, color: np.ndarray, alpha: float = 1.0) -> None:
        a = (mask * alpha)[..., None]
        self.img *= 1 - a
        self.img += a * color

    def shape(self, sdf: np.ndarray, fill: np.ndarray, outline_width: float = 0.0) -> None:
        """Fill a shape, then ink its outline (centred on the edge) when ``outline_width`` > 0."""
        self.paint(self.coverage(sdf), fill)
        if outline_width:
            self.paint(self.coverage(np.abs(sdf) - outline_width / 2), INK)

    def picture(self) -> np.ndarray:
        """The final picture, shrunk to its real size, as 8-bit BGR."""
        small = cv2.resize(np.clip(self.img, 0, 1), (self.w // SUPERSAMPLING, self.h // SUPERSAMPLING),
                           interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(np.round(small * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    def shrink(self, values: np.ndarray) -> np.ndarray:
        """A per-pixel value map at the final size (averaged like the picture)."""
        return cv2.resize(values.astype(np.float32), (self.w // SUPERSAMPLING, self.h // SUPERSAMPLING),
                          interpolation=cv2.INTER_AREA)


def sd_box(x, y, cx: float, cy: float, half_w: float, half_h: float, radius: float = 0.0):
    """Signed distance to a rectangle with rounded corners."""
    qx, qy = np.abs(x - cx) - half_w + radius, np.abs(y - cy) - half_h + radius
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - radius


def sd_circle(x, y, cx: float, cy: float, radius: float):
    return np.hypot(x - cx, y - cy) - radius


# ---------------------------------------------------------------- geometry

def pins() -> List[Tuple[float, float, float, float, Tuple[int, int], int]]:
    """Every pin as (centre x, centre y, half width, half height, outward direction, rank); the rank
    counts from the nearer end of its side (0 = outermost)."""
    cx, cy = CHIP_CENTER
    span = (PIN_COUNT - 1) * PIN_PITCH
    reach = PACKAGE_HALF + PIN_LENGTH / 2 - 2
    out = []
    for i in range(PIN_COUNT):
        offset = -span / 2 + i * PIN_PITCH
        rank = min(i, PIN_COUNT - 1 - i)
        across, along = PIN_WIDTH / 2, PIN_LENGTH / 2
        out += [(cx + offset, cy - reach, across, along, (0, -1), rank),
                (cx + offset, cy + reach, across, along, (0, 1), rank),
                (cx - reach, cy + offset, along, across, (-1, 0), rank),
                (cx + reach, cy + offset, along, across, (1, 0), rank)]
    return out


def network_nodes() -> List[List[Point]]:
    """The centre of every node, layer by layer, spread evenly between NETWORK_TOP and NETWORK_BOTTOM."""
    nodes = []
    for x, count in LAYERS:
        gap = (NETWORK_BOTTOM - NETWORK_TOP) / (count + 1)
        nodes.append([(x, NETWORK_TOP + gap * (i + 1)) for i in range(count)])
    return nodes


def trace_layout(rng: np.random.Generator) -> Tuple[List[Segment], List[Point]]:
    """The board's traces (line segments) and the vias they end in.

    On the top, left and bottom, the traces fan out in parallel: outer pins bend first, so no two
    traces cross, and the straight part grows only as much as the room to the picture's edge allows.
    On the right, the traces carry the chip's output into the network's first layer.
    """
    cx, cy = CHIP_CENTER
    segments, vias, right = [], [], []
    for (px, py, half_w, half_h, (dx, dy), rank) in pins():
        x0, y0 = px + dx * half_w, py + dy * half_h  # the pin's tip
        if dx == 1:
            right.append((x0, y0))
            continue
        away = 1 if ((px - cx) if dy else (py - cy)) > 0 else -1  # bend away from the middle of the side
        room = {(-1, 0): x0, (0, -1): y0, (0, 1): HEIGHT - y0}[(dx, dy)]
        spare = room - TRACE_FIRST_STRAIGHT - TRACE_BEND - TRACE_END[1] - VIA_CLEARANCE
        step = float(np.clip(spare / (PIN_COUNT // 2 - 1), 0, PIN_PITCH))
        straight = TRACE_FIRST_STRAIGHT + rank * step
        x1, y1 = x0 + dx * straight, y0 + dy * straight
        x2, y2 = (x1 + away * TRACE_BEND, y1 + dy * TRACE_BEND) if dy else (x1 + dx * TRACE_BEND, y1 + away * TRACE_BEND)
        end = rng.uniform(*TRACE_END)
        x3, y3 = x2 + dx * end, y2 + dy * end
        segments += [((x0, y0), (x1, y1)), ((x1, y1), (x2, y2)), ((x2, y2), (x3, y3))]
        vias.append((x3 + dx * 4, y3 + dy * 4))
    inputs = network_nodes()[0]
    right.sort(key=lambda tip: tip[1])
    for i, (x0, y0) in enumerate(right):
        nx, ny = inputs[i * len(inputs) // len(right)]
        xm = x0 + 10 + 4 * min(i, len(right) - 1 - i)
        rise = ny - y0
        x2 = min(xm + abs(rise), nx - 22)
        y2 = y0 + np.sign(rise) * (x2 - xm)
        segments += [((x0, y0), (xm, y0)), ((xm, y0), (x2, y2)), ((x2, y2), (nx - 10, ny))]
    return segments, vias


# ---------------------------------------------------------------- layers of the picture

def _sky(p: Painter) -> None:
    t = p.y / HEIGHT
    for (t0, c0), (t1, c1) in zip(SKY, SKY[1:]):
        k = np.clip((t - t0) / (t1 - t0), 0, 1)[..., None]
        inside = ((t >= t0) & (t <= t1))[..., None]
        p.img[:] = np.where(inside, c0 * (1 - k) + c1 * k, p.img)
    gx = np.abs(((p.x + GRID_STEP / 2) % GRID_STEP) - GRID_STEP / 2)
    gy = np.abs(((p.y + GRID_STEP / 2) % GRID_STEP) - GRID_STEP / 2)
    p.paint(p.coverage(np.minimum(gx, gy) - 0.5), WHITE, 0.10)  # grid lines
    p.paint(p.coverage(np.hypot(gx, gy) - 1.4), WHITE, 0.35)    # a dot on every crossing


def _sparkles(p: Painter, rng: np.random.Generator) -> None:
    segments = []
    for _ in range(SPARKLES):
        x, y, r = rng.uniform(0, WIDTH), rng.uniform(0, HEIGHT), rng.uniform(2.5, 6)
        segments += [((x - r, y), (x + r, y)), ((x, y - r), (x, y + r))]
    p.paint(p.lines(segments, 1.4), WHITE, 0.9)


def _plate(p: Painter, text: str, left: float, top: float, scale: float = 0.36) -> None:
    """A label on a light plate with an ink border."""
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 3.5
    p.shape(sd_box(p.x, p.y, left + w / 2 + pad, top + h / 2 + pad, w / 2 + pad, h / 2 + pad, 3), PLATE, 1.6)
    p.paint(p.text(text, (left + pad, top + pad + h), scale), INK)


def _network(p: Painter, rng: np.random.Generator) -> None:
    nodes = network_nodes()
    dim, lit = [], []
    for left, right in zip(nodes, nodes[1:]):
        for a in left:
            for b in right:
                (lit if rng.random() < 0.16 else dim).append((a, b))
    p.paint(p.lines(dim, 1.3), INK, 0.55)
    p.paint(p.lines(lit, 4.2), INK)
    p.paint(p.lines(lit, 1.8), WHITE)
    for li, layer in enumerate(nodes):
        r = 13 if li == len(nodes) - 1 else 10
        for ni, (x, y) in enumerate(layer):
            p.shape(sd_circle(p.x, p.y, x, y, r), NODE_COLORS[(li + ni) % 3 == 0], 2.6)
            p.paint(p.coverage(sd_circle(p.x, p.y, x - r * 0.3, y - r * 0.3, r * 0.32)), WHITE, 0.85)  # highlight
            p.paint(p.coverage(sd_circle(p.x, p.y, x, y, r * 0.3)), INK)
    for (x, y), label in zip(nodes[-1], OUTPUTS):
        _plate(p, label, x - 34, y + 20)
    for layer in nodes[1:-1]:  # a few weights, like a diagram in a paper
        for (x, y) in layer[1::2]:
            _plate(p, f'{rng.uniform(-1, 1):+.2f}', x + 13, y - 22, 0.32)


def _traces(p: Painter, rng: np.random.Generator) -> None:
    segments, vias = trace_layout(rng)
    p.paint(p.lines(segments, 5.0), INK)  # ink first, the trace's colour on top
    p.paint(p.lines(segments, 2.2), TRACE)
    for (x, y) in vias:
        p.shape(sd_circle(p.x, p.y, x, y, 4.6), TRACE, 2.0)
        p.paint(p.coverage(sd_circle(p.x, p.y, x, y, 1.6)), INK)


def _data_matrix(p: Painter, rng: np.random.Generator, left: float, top: float, modules: int = 14,
                 size: float = 2.0) -> None:
    """A Data Matrix-like code, as printed on real chips: a solid L on the left and bottom, alternating
    modules on the top and right, random modules inside."""
    cells = rng.random((modules, modules)) < 0.5
    cells[:, 0] = True
    cells[-1, :] = True
    cells[0, :] = np.arange(modules) % 2 == 0
    cells[:, -1] = np.arange(modules) % 2 == 1
    mask = np.zeros((p.h, p.w), np.float32)
    for row, col in zip(*np.nonzero(cells)):
        x, y = left + (col + 0.5) * size, top + (row + 0.5) * size
        mask = np.maximum(mask, p.coverage(np.maximum(np.abs(p.x - x), np.abs(p.y - y)) - size / 2))
    p.paint(mask, PACKAGE_TEXT)


def _chip(p: Painter, rng: np.random.Generator) -> None:
    x, y = p.x, p.y
    cx, cy = CHIP_CENTER
    package = sd_box(x, y, cx, cy, PACKAGE_HALF, PACKAGE_HALF, PACKAGE_RADIUS)
    p.paint(p.coverage(sd_box(x, y, cx + 8, cy + 10, PACKAGE_HALF, PACKAGE_HALF, PACKAGE_RADIUS)), INK, 0.35)  # shadow
    for (px, py, half_w, half_h, (dx, dy), _rank) in pins():
        pin = sd_box(x, y, px, py, half_w, half_h, 1.2)
        p.shape(pin, GOLD, 2.0)
        # The lit half: the left half of a top or bottom pin, the upper half of a side pin.
        lit = (sd_box(x, y, px - half_w / 2, py, half_w / 2, half_h) if dy else
               sd_box(x, y, px, py - half_h / 2, half_w, half_h / 2))
        p.paint(p.coverage(np.maximum(lit, pin + 1.2)), GOLD_LIT, 0.9)
    # Package: a bevelled frame, lit on the top and left, shaded on the bottom and right.
    p.shape(package, PACKAGE)
    top_face = sd_box(x, y, cx, cy, PACKAGE_HALF - BEVEL, PACKAGE_HALF - BEVEL, PACKAGE_RADIUS - 4)
    frame = p.coverage(package) * (1 - p.coverage(top_face))
    lit = ((x - cx) + (y - cy) < 0).astype(np.float32)
    p.paint(frame * lit, PACKAGE_LIT)
    p.paint(frame * (1 - lit), PACKAGE_SHADE)
    p.paint(p.coverage(np.abs(top_face) - 0.8), INK, 0.6)
    p.paint(p.coverage(np.abs(package) - 1.6), INK)
    # Pin 1 mark, markings and the Data Matrix code.
    p.shape(sd_circle(x, y, cx - PACKAGE_HALF + 26, cy - PACKAGE_HALF + 26, 5.5), PACKAGE_SHADE, 1.6)
    p.paint(p.text('NPU-7', (cx + 22, cy - PACKAGE_HALF + 32), 0.5, 1.4), PACKAGE_TEXT)
    p.paint(p.text('TENSOR CORE  N7  2611', (cx - PACKAGE_HALF + 20, cy + PACKAGE_HALF - 18), 0.36), PACKAGE_TEXT)
    _data_matrix(p, rng, cx - PACKAGE_HALF + 16, cy - 14)
    # Die: a dark screen with a grid of glowing cores and a bright rim.
    die = sd_box(x, y, cx, cy, DIE_HALF, DIE_HALF, 8)
    p.shape(die, DIE)
    gx = np.abs(((x - cx) % CORE_CELL) - CORE_CELL / 2)
    gy = np.abs(((y - cy) % CORE_CELL) - CORE_CELL / 2)
    p.paint(p.coverage(np.maximum(gx, gy) - 3.0) * p.coverage(die + 6), DIE_CORE, 0.28)
    p.paint(p.coverage(np.abs(die + 3.5) - 1.0), DIE_RIM)
    p.paint(p.coverage(np.abs(die) - 1.4), INK)
    # "AI": white strokes with an ink outline, over a pink offset shadow.
    h, ax = 25, cx - 14
    strokes = [((ax, cy - h), (ax - 17, cy + h)), ((ax, cy - h), (ax + 17, cy + h)),
               ((ax - 9, cy + 9), (ax + 9, cy + 9)), ((cx + 25, cy - h), (cx + 25, cy + h))]
    shadow = [((a[0] + 3, a[1] + 3), (b[0] + 3, b[1] + 3)) for a, b in strokes]
    p.paint(p.lines(shadow, 11), INK)
    p.paint(p.lines(shadow, 7), LETTER_SHADOW)
    p.paint(p.lines(strokes, 12), INK)
    p.paint(p.lines(strokes, 7.5), WHITE)


def _depth(p: Painter) -> np.ndarray:
    """NEAR on the chip and its pins, then farther with the distance from them, up to FAR."""
    cx, cy = CHIP_CENTER
    distance = sd_box(p.x, p.y, cx, cy, PIN_REACH, PIN_REACH, 8)
    return p.shrink(np.clip(distance * DEPTH_PER_PIXEL, NEAR, FAR)).round().astype(np.uint8)


def ai_chip_scene(rng: np.random.Generator) -> Canvas:
    """The web demo's sample picture (WIDTH x HEIGHT) and its depth map."""
    p = Painter(WIDTH, HEIGHT)
    _sky(p)
    _sparkles(p, rng)
    _network(p, rng)
    _traces(p, rng)
    _chip(p, rng)
    text, origin = CAPTION
    p.paint(p.text(text, origin, 0.52, 1.3), WHITE)
    canvas = Canvas(WIDTH, HEIGHT)
    canvas.img, canvas.depth = p.picture(), _depth(p)
    return canvas
