# License disclaimer: Docstrings derived from 2.2.0 of TensorFlow, licensed under Apache-2.0

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, AnyStr, Any, Generator, Container

import numpy as np
import tensorflow as tf


class BaseKeras(ABC):
    @abstractmethod
    def compile(self,
                optimizer='rmsprop',
                loss=None,
                metrics=None,
                loss_weights=None,
                weighted_metrics=None,
                sample_weight_mode=None,
                run_eagerly=None,
                **kwargs
                ):
        """
        Configures the model for training.

        :param optimizer: Name of optimizer or optimizer instance.
        :type optimizer: ```AnyStr or tf.keras.optimizers.Optimizer```

        :param loss: Name of objective function or loss instance.
        :type loss: ```Optional[tf.keras.losses.Loss or AnyStr]```

        :param metrics: List of metrics to be evaluated by the model during training
            and testing. Each of this can be a string (name of a built-in function), function
            or a `tf.keras.metrics.Metric` instance.
        :type metrics: ```Optional[List[tf.keras.metrics.Metric or AnyStr]]```

        :param loss_weights: Optional list or dictionary specifying scalar
            coefficients (Python floats) to weight the loss contributions
            of different model outputs.
        :type loss_weights: ```Optional[List[float] or Dict[AnyStr, float]]```

        :param sample_weight_mode: If you need to do timestep-wise
            sample weighting (2D weights), set this to `"temporal"`.
            `None` defaults to sample-wise weights (1D).
            If the model has multiple outputs, you can use a different
            `sample_weight_mode` on each output by passing a
            dictionary or a list of modes.
        :type sample_weight_mode: ```Optional[AnyStr or List[AnyStr] or Dict[str, str]]```

        :param weighted_metrics: List of metrics to be evaluated and weighted
            by sample_weight or class_weight during training and testing.
        :type weighted_metrics: ```Optional[List[tf.keras.metrics.Metric or AnyStr]]```

        :param run_eagerly: eager execution
        :type run_eagerly: ```Optional[Bool]```

        :param kwargs: Any additional arguments
        :type kwargs: ```Optional[Dict[AnyStr, Any]]```

        :raises
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """

    @abstractmethod
    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        :param x: Input data
        :type x: ```List[tf.Tensor or tf.TensorArray or np.ndarray] or tf.TensorArray or np.ndarray
                    or tf.data.Dataset or tf.keras.utils.Sequence or Generator```
        
        :param y: Target data
        :type y: ```List[tf.Tensor or tf.TensorArray or np.ndarray] or tf.TensorArray or np.ndarray
                    or tf.data.Dataset or tf.keras.utils.Sequence or Generator or None```

        :param batch_size: Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        :type batch_size: ```Optional[int]```

        :param epochs: Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        :type epochs: ```int```

        :param verbose: Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).
        :type verbose: None or 0 or 1 or 2

        :param callbacks: List of callbacks to apply during training.
        :type callbacks: ```Optional[List[tf.keras.callbacks]]```

        :param validation_split: Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
           `keras.utils.Sequence` instance.
        :type validation_split: ```float``` between 0 and 1

        :param validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
        :type validation_data: ```Tuple[np.ndarray, np.ndarray] or Tuple[tf.Tensor, tf.Tensor]
                                  or Tuple[np.ndarray, np.ndarray, np.ndarray] or tf.dataset.Dataset```

        :param shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch'). This argument is ignored
            when `x` is a generator. 'batch' is a special option for dealing
            with the limitations of HDF5 data; it shuffles in batch-sized
            chunks. Has no effect when `steps_per_epoch` is not `None`.
        :type shuffle: ```Optional[bool or 'batch']```

        :param class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        :type class_weight: ```Optional[Dict[int, float]]```

        :param sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        :type sample_weight: ```Optional[np.ndarray]```

        :param initial_epoch: Epoch at which to start training
        :type initial_epoch: ```int```

        :param steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            When passing an infinitely repeating dataset, you must specify the
            `steps_per_epoch` argument. This argument is not supported with
            array inputs.
        :type steps_per_epoch: ```Optional[int]```

        :param validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of an infinitely repeated dataset, it will run into an
            infinite loop. If 'validation_steps' is specified and only part of
            the dataset will be consumed, the evaluation will start from the
            beginning of the dataset at each epoch. This ensures that the same
            validation samples are used every time.
        :type validation_steps: ```Optional[int]```

        :param validation_batch_size: Number of samples per validation batch.
            If unspecified, will default to `batch_size`.
            Do not specify the `validation_batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        :type validation_batch_size: ```Optional[int]```

        :param validation_freq: Only relevant if validation data is provided. Integer
            or `collections_abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        :type validation_freq: ```Optional[Container[int] or List[int] or Tuple[int] or int]```

        :param max_queue_size: Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        :type max_queue_size: ```int```

        :param workers: Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1. If 0, will execute the generator on the main
            thread.
        :type workers: ```int```

        :param use_multiprocessing: Used for generator or
            `tf.keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.
        :type use_multiprocessing: ```bool```

        :return: A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).
        :rtype: ```tf.keras.callbacks.History```

        :raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
        """


del ABC, abstractmethod, List, Dict, Optional, AnyStr, Any, Generator, Container, np, tf

__all__ = ['BaseKeras']
