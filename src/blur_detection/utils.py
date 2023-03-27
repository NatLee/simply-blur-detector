"""
Data tools

Date: 2023-03-27

Author: Nat Lee
"""

from pathlib import Path
from loguru import logger
from tqdm import tqdm
import numpy as np
import cv2
import tensorflow as tf

from matplotlib import pyplot as plt
from blurgenerator import motion_blur, lens_blur, gaussian_blur

def check_normal(img: np.ndarray) -> np.ndarray:
    '''Check img is normalized.'''
    if img.dtype == np.uint8 or np.sum(img)/np.prod(img.shape) > 1.:
        img = img / 255.
    return img

def equalize_hist(img: np.ndarray) -> np.ndarray:
    '''Get equalize histogram.'''
    b = cv2.equalizeHist(img[:, :, 0])
    g = cv2.equalizeHist(img[:, :, 1])
    r = cv2.equalizeHist(img[:, :, 2])
    img = np.concatenate((
        b[:, :, np.newaxis],
        g[:, :, np.newaxis],
        r[:, :, np.newaxis]),
        axis=-1
    )
    return img

def preprocessing(img: np.ndarray, input_shape=(256, 256)) -> np.ndarray:
    '''Image preprocessing.'''
    img = cv2.resize(img, input_shape)
    img = check_normal(img)
    return img

def get_data(data_path: str, exts=['.jpg', '.png', '.jpeg'], batch_size=4, seed=87) -> (tf.data.Dataset, tf.data.Dataset):
    '''Get data from paths.'''
    img_paths = [p for p in Path(data_path).rglob('*') if p.suffix in exts]
    x_train = list()
    y_train = list()
    x_test = list()
    y_test = list()

    for img_path in tqdm(img_paths, desc='Preparing images...'):
        img = cv2.imread(img_path.as_posix())
        img = equalize_hist(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = preprocessing(img)

        if img_path.parent.parent.name == 'train':
            x_train.append(img)
            if img_path.parent.name == 'blur':
                y_train.append(1)
            elif img_path.parent.name == 'nonblur':
                y_train.append(0)
        elif img_path.parent.parent.name == 'test':
            x_test.append(img)
            if img_path.parent.name == 'blur':
                y_test.append(1)
            elif img_path.parent.name == 'nonblur':
                y_test.append(0)

    x_train = np.asarray(x_train)
    y_train = tf.keras.utils.to_categorical(np.asarray(y_train), 2)

    x_test = np.asarray(x_test)
    y_test = tf.keras.utils.to_categorical(np.asarray(y_test), 2)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(seed).batch(batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_ds, test_ds

def predict_blur_img(detection_model, img_path:str, blur_mode=None, save_fig=True, save_fig_path='./results.png', show_plot=False, plot_transparent=False):
    '''Run test with generating blur image.
        -------
        img_path: string path
        blur_model: motion, lens, and default is gaussian

        For example:
            predict_blur_img(detection_model, img, blur_mode='lens')
    '''

    img = cv2.imread(Path(img_path).as_posix())
    eq_img = equalize_hist(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    eq_img = cv2.cvtColor(eq_img, cv2.COLOR_BGR2RGB)
    nrows = 4
    ncols = 4
    figsize = (6, 6)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    def preprocess_images(img, nrows, ncols):
        imgs = []
        for idx in range(nrows):
            img_ = []
            for jdx in range(ncols):
                if blur_mode == 'motion':
                    size = (idx+1)*(jdx*2+1)*10
                    angle = (idx+1)*(jdx*2)
                    blur_result = motion_blur(img, size=size, angle=angle)
                elif blur_mode == 'lens':
                    radius = (idx+1)*(jdx*2+1)*2
                    components = 4
                    exposure_gamma = 2
                    blur_result = lens_blur(img, radius=radius, components=components, exposure_gamma=exposure_gamma)
                else:
                    kernel = (idx+1)*(jdx*2+1)*3
                    blur_result = gaussian_blur(img, kernel)
                blur_result = preprocessing(blur_result)
                img_.append(blur_result)
            imgs.append(img_)
        return imgs

    imgs = preprocess_images(img, nrows, ncols)
    eq_imgs = preprocess_images(eq_img, nrows, ncols)

    for idx, axe in enumerate(axes):
        for jdx, ax in enumerate(axe):
            blur_result = eq_imgs[idx][jdx]
            result, text = detection_model.predict(blur_result)
            logger.info(result)
            logger.info(text)
            #ax.set_title(text[0], color='white')
            ax.set_title(text[0])
            ax.axis('off')
            origin_blur_result = imgs[idx][jdx]
            ax.imshow(origin_blur_result)
            ax.grid(False)

    plt.tight_layout()
    if save_fig:
        fig.savefig(save_fig_path, transparent=plot_transparent)
    if show_plot:
        plt.show() # no need to show in container
