"""
Data tools: load a data folder for training, and draw the comparison figure of a prediction.

Date: 2023-03-27

Author: Nat Lee
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import keras
import numpy as np
import tensorflow as tf
from loguru import logger
from matplotlib import pyplot as plt
from tqdm import tqdm

from .blur import ladder_blur
from .config import (
    BLUR_CLASS,
    CLASS_NAMES,
    DEMO_FIGURE_PATH,
    FIGURE_GRID,
    FIGURE_SIZE,
    IMAGE_EXTENSIONS,
    INPUT_SIZE,
    LABEL_FOLDERS,
    SPLIT_FOLDERS,
)
# check_normal, equalize_hist and preprocessing used to live here; importing them from
# blur_detection.utils keeps working.
from .images import check_normal, equalize_hist, prepare_image, preprocessing, random_crops, read_image

__all__ = [
    'check_normal', 'equalize_hist', 'preprocessing', 'prepare_image', 'read_image',
    'find_images', 'load_images', 'get_data', 'predict_blur_img',
]


def find_images(data_path: str, exts: Sequence[str] = IMAGE_EXTENSIONS) -> Dict[str, List[Tuple[Path, int]]]:
    """List the labelled images of each split: ``{'train': [(path, label), ...], 'test': [...]}``.

    Images must sit in ``<data_path>/<train|test>/<blur|nonblur>/``; other files and folders are
    skipped. Extensions are compared case-insensitively. Paths are sorted, so the order is the same on
    every machine.
    """
    exts = {ext.lower() for ext in exts}
    found: Dict[str, List[Tuple[Path, int]]] = {split: [] for split in SPLIT_FOLDERS}
    for path in sorted(Path(data_path).rglob('*')):
        split, label = path.parent.parent.name, LABEL_FOLDERS.get(path.parent.name)
        if path.suffix.lower() in exts and split in found and label is not None:
            found[split].append((path, label))
    return found


def load_images(samples: Sequence[Tuple[Path, int]], desc: str = 'Preparing images...', crops: int = 0,
                rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Read and prepare labelled images: model inputs (N x 256 x 256 x 3) and one-hot labels (N x 2).

    With ``crops``, every image also adds that many random square crops (see `random_crops`).
    """
    rng = rng if rng is not None else np.random.default_rng()
    inputs, labels = [], []
    for path, label in tqdm(samples, desc=desc):
        img = read_image(path)
        views = [img, *random_crops(img, crops, rng)]
        inputs.extend(prepare_image(view) for view in views)
        labels.extend([label] * len(views))
    return np.stack(inputs), keras.utils.to_categorical(labels, len(CLASS_NAMES))


def get_data(data_path: str, exts: Sequence[str] = IMAGE_EXTENSIONS, batch_size: int = 4,
             seed: int = 87, crops: int = 0) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Get the training and test datasets from a data folder (see `find_images` for the layout).

    The training images are shuffled again every epoch; ``seed`` makes the order (and the crops)
    repeatable. ``crops`` adds that many random crops of every training image, which teaches the
    model to judge pieces of a picture too (the web demo's heat map needs that); 0 trains on whole
    images only, as before.
    """
    found = find_images(data_path, exts)
    empty = [split for split, samples in found.items() if not samples]
    if empty:
        folders = ' and '.join(f'{data_path}/{split}/{{{",".join(LABEL_FOLDERS)}}}/' for split in empty)
        raise ValueError(f'No images found in {folders}.')

    x_train, y_train = load_images(found['train'], crops=crops, rng=np.random.default_rng(seed))
    x_test, y_test = load_images(found['test'])
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train), seed=seed).batch(batch_size)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_ds, test_ds


def predict_blur_img(detection_model, img_path: str, blur_mode: Optional[str] = None, save_fig: bool = True,
                     save_fig_path: str = DEMO_FIGURE_PATH, show_plot: bool = False,
                     plot_transparent: bool = False) -> List[List[Tuple[str, float]]]:
    '''Blur an image more and more, predict every step and draw the results as a grid.

        -------
        img_path: string path
        blur_mode: motion, lens, gaussian, and default (None) is gaussian

        Every tile shows the blurred image with the predicted label and the probability that it is
        blurry. Returns the same as a grid of ``(label, p_blur)``.

        For example:
            predict_blur_img(detection_model, img, blur_mode='lens')
    '''
    img = read_image(img_path)
    nrows, ncols = FIGURE_GRID
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=FIGURE_SIZE, squeeze=False)

    results = []
    for idx, axe in enumerate(axes):
        row = []
        for jdx, ax in enumerate(axe):
            blur_result = ladder_blur(img, blur_mode, idx, jdx)
            result, text = detection_model.predict(prepare_image(blur_result))
            p_blur = float(np.asarray(result)[0, BLUR_CLASS])
            logger.info(f'tile {idx},{jdx}: {text[0]} (P(blur) = {p_blur:.3f})')
            row.append((text[0], p_blur))

            ax.set_title(f'{text[0]} {p_blur:.0%}')
            ax.axis('off')
            shown = preprocessing(cv2.cvtColor(blur_result, cv2.COLOR_BGR2RGB), (INPUT_SIZE, INPUT_SIZE))
            ax.imshow(shown)
            ax.grid(False)
        results.append(row)

    plt.tight_layout()
    if save_fig:
        Path(save_fig_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_fig_path, transparent=plot_transparent)
        logger.info(f'Saved the comparison figure to {save_fig_path}')
    if show_plot:
        plt.show()  # no need to show in container
    plt.close(fig)
    return results
