"""
Command line: ``blur-detector <command> [--flag=value ...]``.

    draw      draw the built-in data set: sharp pictures made from code (data/*/nonblur)
    generate  write a blurred copy of every sharp image (data/*/nonblur -> data/*/blur)
    train     train the model on the data folder and save it
    predict   predict one image and draw the comparison figure
    export    export the model for the web demo (ONNX + model card)
    run       train, save and draw the comparison figure (what main.py used to do)

``blur-detector <command> --help`` lists the flags of a command.
"""
from typing import Optional

import fire
import keras
import numpy as np
from loguru import logger

from .blur import generate_blur_images
from .config import (
    BLUR_CLASS,
    DATA_DIR,
    DATASET_SEED,
    DATASET_SIZES,
    DEFAULT_EPOCHS,
    DEMO_BLUR_MODE,
    DEMO_FIGURE_PATH,
    DEMO_IMAGE,
    MODEL_PATH,
    WEB_DIR,
)
from .detection_model import DetectionModel
from .export import export_web_model
from .images import prepare_image, read_image
from .scenes import draw_dataset
from .utils import get_data, predict_blur_img


def draw(data_path: str = DATA_DIR, train: int = DATASET_SIZES['train'], test: int = DATASET_SIZES['test'],
         seed: int = DATASET_SEED) -> None:
    """Draw the built-in data set: sharp pictures made from code, in <data_path>/{train,test}/nonblur.

    Also draws the demo picture <data_path>/demo.png. The default seed gives exactly the pictures in
    the repository's data/ folder; run `generate` afterwards for the blurred copies.
    """
    written = draw_dataset(data_path, {'train': train, 'test': test}, seed=seed)
    logger.info(f'Drew {len(written)} pictures.')


def generate(data_path: str = DATA_DIR, seed: Optional[int] = None) -> None:
    """Write a blurred copy of every image in <data_path>/*/nonblur to the blur folder next to it.

    Each image gets a lens, Gaussian or motion blur of random strength. Give a seed to repeat a run.
    """
    written = generate_blur_images(data_path, seed=seed)
    logger.info(f'Wrote {len(written)} blurred images.')


def _load(model_path: Optional[str], seed: Optional[int]) -> DetectionModel:
    """Load the model (or start an untrained one); a seed makes the starting weights repeatable."""
    if seed is not None:
        keras.utils.set_random_seed(seed)
    return DetectionModel(model_dir=model_path)


def _fit(detector: DetectionModel, data_path: str, epochs: int, batch_size: int = 4, crops: int = 0) -> None:
    train_ds, test_ds = get_data(data_path, batch_size=batch_size, crops=crops)
    detector.train(train_ds, test_ds, epochs=epochs)


def _save(detector: DetectionModel, model_save_path: str) -> None:
    detector.save(model_save_path)
    logger.info(f'Saved the model to {model_save_path}')


def train(data_path: str = DATA_DIR, model_path: Optional[str] = MODEL_PATH,
          model_save_path: str = MODEL_PATH, epochs: int = DEFAULT_EPOCHS, batch_size: int = 4,
          seed: Optional[int] = None, crops: int = 0) -> None:
    """Train on <data_path>/{train,test}/{blur,nonblur} and save the model to model_save_path.

    Training continues from model_path when that file exists; pass --model_path=None to start over.
    A seed makes the starting weights repeatable. --crops=N also trains on N random crops of every
    training image, so the model can judge parts of a picture (the web demo's heat map).
    """
    detector = _load(model_path, seed)
    _fit(detector, data_path, epochs, batch_size, crops)
    _save(detector, model_save_path)


def predict(img_path: str = DEMO_IMAGE, model_path: str = MODEL_PATH, blur_mode: Optional[str] = DEMO_BLUR_MODE,
            save_fig_path: str = DEMO_FIGURE_PATH) -> float:
    """Predict one image, then blur it more and more and draw the predictions as a 4 x 4 grid.

    blur_mode is lens, gaussian or motion (None means gaussian). Returns P(blur) of the image as is.
    """
    detector = DetectionModel(model_dir=model_path)
    result, text = detector.predict(prepare_image(read_image(img_path)))
    p_blur = float(np.asarray(result)[0, BLUR_CLASS])
    logger.info(f'{img_path}: {text[0]} (P(blur) = {p_blur:.3f})')
    predict_blur_img(detector, img_path, blur_mode=blur_mode, save_fig_path=save_fig_path)
    return p_blur


def export(model_path: str = MODEL_PATH, data_path: str = DATA_DIR, web_dir: str = WEB_DIR) -> str:
    """Export the model into the web demo folder (<web_dir>/model) with a model card.

    The card records the accuracy on <data_path>, the package versions and, in GitHub Actions, the
    commit and workflow run that made it.
    """
    detector = DetectionModel(model_dir=model_path)
    return str(export_web_model(detector.model, data_path, web_dir))


def run(model_path: str = MODEL_PATH, model_save_path: str = MODEL_PATH, data_path: str = DATA_DIR,
        train_model: bool = True, epochs: int = DEFAULT_EPOCHS, save_model: bool = True, demo: bool = True,
        demo_img_path: str = DEMO_IMAGE, demo_blur_mode: Optional[str] = DEMO_BLUR_MODE,
        seed: Optional[int] = None) -> None:
    """Train (and save) the model, then draw the comparison figure of demo_img_path.

    Same flags and defaults as the original main.py; seed is new and optional.
    """
    detector = _load(model_path, seed)
    if train_model:
        _fit(detector, data_path, epochs)
        if save_model:
            _save(detector, model_save_path)
    if demo:
        predict_blur_img(detector, demo_img_path, blur_mode=demo_blur_mode)


COMMANDS = {
    'draw': draw,
    'generate': generate,
    'train': train,
    'predict': predict,
    'export': export,
    'run': run,
}


def main() -> None:
    fire.Fire(COMMANDS, name='blur-detector')
