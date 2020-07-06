""" Implementation of ml_params API """

# Mostly based off https://github.com/keras-team/keras-io/blob/8320a6c/examples/vision/mnist_convnet.py

from os import path
from sys import stdout
from typing import Tuple

import numpy as np
import tensorflow as tf
from ml_params.base import BaseTrainer

from ml_params_keras import get_logger
from ml_params_keras.datasets import load_data_from_keras_or_ml_prepare

logger = get_logger('.'.join((path.basename(path.dirname(__file__)),
                              path.basename(__file__).rpartition('.')[0])))


class KerasTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Keras """

    data = None  # type: (None or Tuple[tf.data.Dataset, tf.data.Dataset] )
    model = None  # contains the model, e.g., a `tl.Serial`

    def load_data(self, dataset_name, data_loader=load_data_from_keras_or_ml_prepare,
                  data_type='infer', output_type=None, K=None,
                  **data_loader_kwargs):
        """
        Load the data for your ML pipeline. Will be fed into `train`.

        :param dataset_name: name of dataset
        :type dataset_name: ```str```

        :param data_loader: function that returns the expected data type.
         Defaults to TensorFlow Datasets and ml_prepare combined one.
        :type data_loader: ```None or (*args, **kwargs) -> tf.data.Datasets or Any```

        :param data_loader_kwargs: pass this as arguments to data_loader function
        :type data_loader_kwargs: ```**data_loader_kwargs```

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
                                                        data_loader=data_loader,
                                                        data_type=data_type,
                                                        output_type=output_type,
                                                        K=K,
                                                        **data_loader_kwargs)

    def train(self, callbacks, epochs, loss, metrics, metric_emit_freq, optimizer,
              save_directory, output_type='infer', writer=stdout,
              validation_split=0.1, batch_size=128,
              *args, **kwargs):
        """
        Run the training loop for your ML pipeline.

        :param callbacks: Collection of callables that are run inside the training loop
        :type callbacks: ```None or List[Callable] or Tuple[Callable]```

        :param epochs: number of epochs (must be greater than 0)
        :type epochs: ```int```

        :param loss: Loss function, can be a string (depending on the framework) or an instance of a class
        :type loss: ```str or Callable or Any```

        :param metrics: Collection of metrics to monitor, e.g., accuracy, f1
        :type metrics: ```None or List[Callable or str] or Tuple[Callable or str]```

        :param metric_emit_freq: Frequency of metric emission, e.g., `lambda: epochs % 10 == 0`, defaults to every epoch
        :type metric_emit_freq: ```None or (*args, **kwargs) -> bool```

        :param optimizer: Optimizer, can be a string (depending on the framework) or an instance of a class
        :type callbacks: ```str or Callable or Any```

        :param save_directory: Directory to save output in, e.g., weights in h5 files. If None, don't save.
        :type save_directory: ```None or str```

        :param output_type: `if save_directory is not None` then save in this format, e.g., 'h5'.
        :type output_type: ```str```

        :param writer: Writer for all output, could be a TensorBoard instance, a file handler like stdout or stderr
        :type writer: ```stdout or Any```

        :param validation_split:
        :type validation_split: ```float```

        :param batch_size:
        :type batch_size: ```int```

        :param args:
        :param kwargs:
        :return:
        """
        super(KerasTrainer, self).train(callbacks=callbacks,
                                        epochs=epochs,
                                        loss=loss,
                                        metrics=metrics,
                                        metric_emit_freq=metric_emit_freq,
                                        optimizer=optimizer,
                                        save_directory=save_directory,
                                        output_type='infer',
                                        writer=writer,
                                        *args, **kwargs)

        assert self.data is not None
        assert self.model is not None

        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        self.model.fit(self.data[0][0], self.data[0][1],
                       batch_size=batch_size, epochs=epochs,
                       validation_split=validation_split)
        score = self.model.evaluate(self.data[1][0], self.data[1][1], verbose=0)
        print("Test loss:\t\t", score[0],
              "\nTest accuracy:\t", score[1], sep='')  # TODO: Parse out metric name, for multiple

        return self.model


del Tuple, tf, get_logger

__all__ = ['KerasTrainer']
