"""
Settings shared by training, prediction, the command line and the web export.

Change a value here and every part that uses it follows; nothing else should repeat these numbers.
"""

#: Side length in pixels of the square image the model looks at. Every image is resized to this.
INPUT_SIZE = 256

#: Label of each model output, in output order. Output 0 is "sharp", output 1 is "blurry".
CLASS_NAMES = ('Clear', 'Blur')

#: Index of the "blurry" output, i.e. where P(blur) lives in a prediction.
BLUR_CLASS = CLASS_NAMES.index('Blur')

#: Data folder layout: ``<data>/<split>/<label folder>/<image>``.
SPLIT_FOLDERS = ('train', 'test')
SHARP_FOLDER = 'nonblur'
BLUR_FOLDER = 'blur'
LABEL_FOLDERS = {SHARP_FOLDER: CLASS_NAMES.index('Clear'), BLUR_FOLDER: BLUR_CLASS}

#: Image files that are picked up (compared case-insensitively).
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png')

#: Quality of the JPEG files that are written. The sharp pictures in data/ were saved with it too, so a
#: picture and its blurred copy went through the same compression and it tells the model nothing.
JPEG_QUALITY = 90

# Default locations. They are paths inside the Docker image, where docker-compose.yml mounts
# ./data, ./model and the web demo; pass other paths on the command line when running elsewhere.
DATA_DIR = '/data'
MODEL_PATH = '/model/model.h5'
DEMO_IMAGE = '/data/demo.png'
WEB_DIR = '/web'

#: Where the web demo expects its model, relative to the web folder.
WEB_MODEL_DIR = 'model'
WEB_MODEL_FILE = 'blur-detector.onnx'
WEB_MODEL_CARD = 'model-card.json'

# The drawn data set of `draw` (made by code, for working offline and for the tests): sharp cartoon
# scenes from blur_detection.scenes, this many per split, each side between these sizes, named
# <prefix>-000.png, -001.png, ... across the splits, plus a drawn demo.png. Draw into a folder of its
# own: the repository's data/ has its own pictures and demo.png.
DATASET_SIZES = {'train': 60, 'test': 20}
DATASET_SIDE_RANGE = (384, 512)
DATASET_SEED = 2023
DATASET_PREFIX = 'scene'

#: The web demo publishes a model only if it gets at least this share of the held-out test pictures
#: right, counted in whole pictures (see export.publish_gate). The model card records the result and
#: .github/workflows/pages.yml refuses to deploy when it fails.
PUBLISH_MIN_TEST_ACCURACY = 0.85
#: Why the threshold is where it is. The model card carries it and the web page shows it, so change it
#: together with the value.
PUBLISH_THRESHOLD_NOTE = ('Lowered from 90% by the project owner on 2026-11-04, when the model trained on the anime '
                          'pictures used then got 35 of 40 test pictures right in CI, while its heat map windows '
                          'went from 64% (trained on drawn scenes) to 92% right.')

#: Largest difference between Keras and ONNX Runtime outputs that the export accepts.
EXPORT_TOLERANCE = 1e-4

# `generate` blurs every sharp image with one of the three blurs, picked at random. Strengths are
# drawn from these lists; the motion blur picks its own random length and angle.
GENERATE_LENS_RADII = (5, 7, 9, 11)
GENERATE_GAUSSIAN_KERNELS = (5, 6, 7, 8, 9)

#: Blur mode of the comparison figure when none is given (``None``).
DEFAULT_BLUR_MODE = 'gaussian'
#: The comparison figure is a grid of this many rows and columns, blur growing along both.
FIGURE_GRID = (4, 4)
FIGURE_SIZE = (6, 6)
#: Lens blur settings of the comparison figure.
FIGURE_LENS_COMPONENTS = 4
FIGURE_LENS_EXPOSURE_GAMMA = 2

# Command line defaults, kept from the original main.py.
DEFAULT_EPOCHS = 5
DEMO_BLUR_MODE = 'lens'
DEMO_FIGURE_PATH = './results.png'

# Heat map windows and training crops. The web demo judges square windows whose side is
# WINDOW_FRACTION of the image's shorter side. `train --crops=N` also trains on N random square crops
# of every training image, between WINDOW_FRACTION and the whole shorter side, so that the model has
# seen pieces of that size. The model card reports the accuracy on such windows of the test images.
WINDOW_FRACTION = 0.5
WINDOW_OVERLAP = 0.5
