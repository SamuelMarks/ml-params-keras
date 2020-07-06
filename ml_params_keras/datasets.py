from os import path, environ

import numpy as np

if environ.get('TF_KERAS', True):
    from tensorflow import keras
else:
    import keras

from ml_params.datasets import load_data_from_ml_prepare
from ml_prepare.datasets import datasets2classes


def load_data_from_keras_or_ml_prepare(dataset_name, tfds_dir=None, K=None, as_numpy=True, **data_loader_kwargs):
    """
    Acquire from the official keras model zoo, or the ophthalmology focussed ml-prepare library

    :param dataset_name: name of dataset
    :type dataset_name: ```str```

    :param tfds_dir: directory to look for models in. Default is ~/tensorflow_datasets.
    :type tfds_dir: ```None or str```

    :param K: backend engine, e.g., `np` or `tf`
    :type K: ```None or np or tf or Any```

    :param as_numpy: Convert to numpy ndarrays
    :type as_numpy: ```bool```

    :param data_loader_kwargs: pass this as arguments to data_loader function
    :type data_loader_kwargs: ```**data_loader_kwargs```

    :return: Train and tests dataset splits
    :rtype: ```Tuple[tf.data.Dataset, tf.data.Dataset] or Tuple[np.ndarray, np.ndarray]```
    """
    if dataset_name in datasets2classes:
        return load_data_from_ml_prepare(dataset_name=dataset_name,
                                         tfds_dir=tfds_dir,
                                         **data_loader_kwargs)
    data_loader_kwargs.update({
        'dataset_name': dataset_name,
        'tfds_dir': tfds_dir,

    })
    if 'scale' not in data_loader_kwargs:
        data_loader_kwargs['scale'] = 255
    load_data_kwargs = {}
    if not path.isabs(tfds_dir):
        # has to be relative to '~/.keras/datasets' =(
        load_data_kwargs = {'path': path.join(tfds_dir, 'downloads',
                                              '{dataset_name}.npz'.format(dataset_name=dataset_name))}
    (x_train, y_train), (x_test, y_test) = getattr(keras.datasets, dataset_name).load_data(
        **load_data_kwargs
    )
    x_train = x_train.astype("float32") / data_loader_kwargs['scale']
    x_test = x_test.astype("float32") / data_loader_kwargs['scale']
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = keras.utils.to_categorical(y_train, data_loader_kwargs['num_classes'])
    y_test = keras.utils.to_categorical(y_test, data_loader_kwargs['num_classes'])

    return (x_train, y_train), (x_test, y_test)
