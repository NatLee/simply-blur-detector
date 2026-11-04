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
#: Sample images shown by the web demo, relative to the web folder, and their index file.
WEB_SAMPLES_DIR = 'samples'
WEB_SAMPLES_INDEX = 'samples.json'

# The built-in data set (data/README.md): sharp pictures drawn by blur_detection.scenes, then blurred
# copies made by `generate`. `draw` writes this many per split, each side between these sizes, named
# <prefix>-000.png, -001.png, ... across the splits.
DATASET_SIZES = {'train': 60, 'test': 20}
DATASET_SIDE_RANGE = (384, 512)
DATASET_SEED = 2023
DATASET_PREFIX = 'scene'
#: Seed of the blur choices in the built-in data set (`generate --seed`).
DATASET_BLUR_SEED = 1

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
