"""
Example for using blur detector
"""
import fire

from blur_detection import DetectionModel
from blur_detection.utils import predict_blur_img
from blur_detection.utils import get_data

def execute(
    model_path="/model/model.h5",
    model_save_path="/model/model.h5",
    data_path="/data",
    train_model=True,
    epochs=5,
    save_model=True,
    demo=True,
    demo_img_path='/data/Mika.png',
    demo_blur_mode='lens'
    ):

    # Loading model
    detector = DetectionModel(model_dir=model_path)

    # ===============================
    # training
    if train_model:
        train_ds, test_ds = get_data(data_path)
        detector.train(train_ds, test_ds, epochs=epochs)
        if save_model:
            detector.save(model_save_path)
    # ===============================

    # ===============================
    # predicting DEMO
    if demo:
        predict_blur_img(
            detector,
            demo_img_path,
            blur_mode=demo_blur_mode
        )
    # ===============================

if __name__ == "__main__":
    fire.Fire(execute)