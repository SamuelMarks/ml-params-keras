""" Implementation of ml_params API """

# Mostly based off https://github.com/keras-team/keras-io/blob/8320a6c/examples/vision/mnist_convnet.py

from os import path, environ
from typing import Tuple

import numpy as np
import tensorflow as tf
from ml_params.base import BaseTrainer
from ml_params_trax import get_logger
from ml_prepare.datasets import datasets2classes
from ml_prepare.exectors import build_tfds_dataset

if environ.get('TF_KERAS', True):
    from tensorflow import keras
else:
    import keras

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class KerasTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Trax """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] )
    model = None  # contains the model, e.g., a `tl.Serial`

    def __init__(self, model, **model_kwargs):
        super(KerasTrainer, self).__init__()
        self.model = model(**model_kwargs)

    def load_data(self, dataset_name, data_loader=None,
                  data_loader_kwargs=None, data_type='infer',
                  output_type=None, K=None):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :param data_type: incoming data type, defaults to 'infer'
        :type data_type: ```str```

        :param output_type: outgoing data_type, defaults to no conversion
        :type output_type: ```None or 'numpy'```

        :param K: backend engine, e.g., `np` or `tf`
        :type K: ```None or np or tf or Any```

        :return: Dataset splits (by default, your train and test)
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        self.data = super(KerasTrainer, self).load_data(dataset_name=dataset_name,
                                                        data_loader=self.load_data_from_keras_or_ml_prepare,
                                                        data_loader_kwargs=data_loader_kwargs,
                                                        data_type=data_type,
                                                        output_type=output_type)
        # self.data = trax.supervised.Inputs(*self.data)
        # trax.supervised.inputs.dataset_to_stream(self.data, dataset_name)

    @staticmethod
    def load_data_from_keras_or_ml_prepare(dataset_name, tensorflow_datasets_dir=None, data_loader_kwargs=None):
        """
        Acquire from the official keras model zoo, or the ophthalmology focussed ml-prepare library

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param tensorflow_datasets_dir: directory to look for models in. Default is ~/tensorflow_datasets.
        :type tensorflow_datasets_dir: ```None or str```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```None or dict```

        :return: Train and tests dataset splits
        :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
        """
        data_loader_kwargs.update({
            'dataset_name': dataset_name,
            'tfds_dir': tensorflow_datasets_dir,

        })
        if 'scale' not in data_loader_kwargs:
            data_loader_kwargs['scale'] = 255

        if dataset_name in datasets2classes:
            ds_builder = build_tfds_dataset(**data_loader_kwargs)

            if hasattr(ds_builder, 'download_and_prepare_kwargs'):
                download_and_prepare_kwargs = getattr(ds_builder, 'download_and_prepare_kwargs')
                delattr(ds_builder, 'download_and_prepare_kwargs')
            else:
                download_and_prepare_kwargs = None

            return BaseTrainer.common_dataset_handler(
                ds_builder=ds_builder,
                download_and_prepare_kwargs=download_and_prepare_kwargs,
                scale=None, K=None, as_numpy=False
            )
        else:
            (x_train, y_train), (x_test, y_test) = getattr(keras.datasets, dataset_name).load_data(
                path.join(tensorflow_datasets_dir, 'downloads',
                          '{dataset_name}.npz'.format(dataset_name=dataset_name))
                # relative to '~/.keras/datasets' =(
            )
            x_train = x_train.astype("float32") / data_loader_kwargs['scale']
            x_test = x_test.astype("float32") / data_loader_kwargs['scale']
            # Make sure images have shape (28, 28, 1)
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)

            y_train = keras.utils.to_categorical(y_train, data_loader_kwargs['num_classes'])
            y_test = keras.utils.to_categorical(y_test, data_loader_kwargs['num_classes'])

            return (x_train, y_train), (x_test, y_test)

    def train(self, epochs, validation_split=0.1, batch_size=128, *args, **kwargs):
        super(KerasTrainer, self).train(epochs=epochs, *args, **kwargs)
        assert self.data is not None
        assert self.model is not None

        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        self.model.fit(self.data[0][0], self.data[0][1],
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split)
        score = self.model.evaluate(self.data[1][0], self.data[1][1], verbose=0)
        print("Test loss:\t\t", score[0],
              "\nTest accuracy:\t", score[1], sep='')

        return self.model


del Tuple, tf, build_tfds_dataset, get_logger

__all__ = ['KerasTrainer']
