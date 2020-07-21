# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from tensorflow import keras
from brainium.nn.layers import Layer
from brainium.common.generic import content


# ----------------------------------------------------------------------------------------------------------------------
# POOLING
# Progressively reduce the spatial size of the representation to reduce the amount
# of parameters and computation in the network.
# ----------------------------------------------------------------------------------------------------------------------
class Pooling(Layer):
    """
    Progressively reduce the spatial size of the representation to reduce the amount
    of parameters and computation in the network.
    ---------
    @author:    Hieu Tr. Pham.
    @modified:  21st July, 2020.
    """

    # Define available pooling operations.
    OPERATORS = {
        'max': [keras.layers.MaxPool1D, keras.layers.MaxPool2D, keras.layers.MaxPool3D],
        'avg': [keras.layers.AvgPool1D, keras.layers.AvgPool2D, keras.layers.AvgPool3D],
        'global_max': [keras.layers.GlobalMaxPool1D, keras.layers.GlobalMaxPool2D, keras.layers.GlobalMaxPool3D],
        'global_avg': [keras.layers.GlobalAvgPool1D, keras.layers.GlobalAvgPool2D, keras.layers.GlobalAvgPool3D]
    }

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        # Invoke parent constructor.
        super(Pooling, self).__init__(**kwargs)
        # Init pooling method.
        self.method = self.args.pop('method', 'max')

    def build(self, input_shape):
        """
        Build the layer.
        :param input_shape: shape of input tensor.
        :return:            any.
        """
        # Calculate operation index based on dimension.
        dim = len(input_shape) - 2 - 1
        # Verify pooling method.
        assert self.method in Pooling.OPERATORS, self.message('only supports: %s' % content(Pooling.OPERATORS.keys()))
        # Verify pooling dimension.
        assert 0 < dim < len(Pooling.OPERATORS), self.message('%s only supports dimension 1D, 2D and 3D.')
        # Init pooling operation based on method and dimension.
        self.operation = Pooling.OPERATORS[self.method][dim](**self.args)

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments schema.
        """
        # Get method.
        method = kwargs.get('method', 'max')
        # Return schema based on method.
        return dict() if 'global' in method else dict(size=2, strides=2, padding='valid')

    def keymap(self, **kwargs) -> dict:
        """
        Get arguments keymap of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments keymap.
        """
        # Get method.
        method = kwargs.get('method', 'max')
        # Return keymap based on method.
        return dict() if 'global' in method else dict(size='pool_size')

    def title(self) -> str:
        """
        Generate title of layer when summary.
        :return: title of layer.
        """
        return '%s_pooling_%s' % (self.method, self.taxonomy.suffix)

    def detail(self) -> str:
        """
        Generate the details of layer when plotted.
        :return: details of layer.
        """
        return 'method: %s' % self.method