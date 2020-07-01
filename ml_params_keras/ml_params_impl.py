""" Implementation of ml_params API """

from ml_params.base import BaseTrainer


class KerasTrainer(BaseTrainer):
    """ Implementation of ml_params BaseTrainer for Keras """
    data = None

    def load_data(self, dataset_name, data_loader, data_type='tensorflow_datasets', output_type='numpy'):
        self.data = super(KerasTrainer, self).load_data(dataset_name=dataset_name,
                                                       data_loader=data_loader,
                                                       data_type=data_type,
                                                       output_type=output_type)

    def train(self, epochs, *args, **kwargs):

        super(KerasTrainer, self).train(epochs=epochs, *args, **kwargs)
        assert self.data is not None
        raise NotImplementedError()


del BaseTrainer

__all__ = ['KerasTrainer']
