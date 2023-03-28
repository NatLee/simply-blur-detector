"""
Blur detection
"""
from pathlib import Path
from loguru import logger
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Model

class DetectionModelStructure(Model):
    """
    Detection Model Structure
    ...

    Methods
    -------
    @ call(x: np.ndarray)
        Model input.
    """
    def __init__(self):
        super(DetectionModelStructure, self).__init__()
        self.conv1 = Conv2D(128, 3, activation='selu')
        self.pool1 = MaxPool2D((2, 2))
        self.conv2 = Conv2D(64, 3, activation='selu')
        self.pool2 = MaxPool2D((4, 4))
        self.conv3 = Conv2D(32, 3, activation='selu')
        self.pool3 = MaxPool2D((8, 8))
        self.flatten = Flatten()
        self.d1 = Dense(16, activation='selu')
        self.d2 = Dense(2, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.d1(x)
        out = self.d2(x)
        return out

class DetectionModel():
    """
    Detection module
    ...

    Attributes
    ----------
    @ model_dir : str
        model path.

    Methods
    -------
    @ train(train_ds, test_ds, epochs=10)
        Train with blur and non blur data.
    @ predict(input_img)
        Predict images.
    @ save(model_dir)
        Save the weights of model to a path.
    """
    def __init__(self, model_dir:str, input_shape=(None, 256, 256, 3)):
        self.model = DetectionModelStructure()
        self.__loss_object = tf.keras.losses.CategoricalCrossentropy()
        self.__optimizer = tf.keras.optimizers.Adam()

        self.__train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.__train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.__test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.__test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        self.model.compile(loss=self.__train_loss, optimizer=self.__optimizer)
        self.model.build(input_shape)

        if Path(model_dir).is_file():
            self.model.load_weights(model_dir)
            logger.info('Blur Detection Model Loaded.')
        else:
            logger.warning('Cannot find blur detection model. Please check model directory.')

    @tf.function
    def __train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images)
            loss = self.__loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.__optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.__train_loss(loss)
        self.__train_accuracy(labels, predictions)

    @tf.function
    def __test_step(self, images, labels):
        predictions = self.model(images)
        t_loss = self.__loss_object(labels, predictions)
        self.__test_loss(t_loss)
        self.__test_accuracy(labels, predictions)

    def save(self, model_dir: str) -> bool:
        '''Save model weights.'''
        self.model.save_weights(model_dir)
        return True

    def train(self, train_ds, test_ds, epochs=10):
        '''Training!'''
        for epoch in range(epochs):
            self.__train_loss.reset_states()
            self.__train_accuracy.reset_states()

            for images, labels in train_ds:
                self.__train_step(images, labels)

            for test_images, test_labels in test_ds:
                self.__test_step(test_images, test_labels)
            logger.info(f'Epoch {epoch+1} / {epochs}')
            desc = 'Loss: {:.5f}, Acc: {:.2f} % | Test Loss: {:.5f}, Test Acc: {:.2f} %'
            logger.info(desc.format(self.__train_loss.result(),
                              self.__train_accuracy.result()*100,
                              self.__test_loss.result(),
                              self.__test_accuracy.result()*100))

    def predict(self, imgs):
        '''Predict!'''
        if isinstance(imgs, np.ndarray):
            imgs = np.expand_dims(imgs, axis=0)
        result = self.model(imgs)
        text = ['Clear' if np.argmax(result) == 0 else 'Blur']
        return result, text
