# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from copy import copy
from tensorflow import keras
from brainium.nn.layers import Layer
from brainium.common import KwargParse, generic


# Default pooling arguments parser.
PARSER = KwargParse().add('data_format', None, keras.backend.image_data_format())


class Pooling(Layer):
    """
    Reduce the spatial size of the representation by keep max elements in pooling window.
    ---------
    @author:    Hieu Pham.
    @created:   13rd August, 2020.
    """
    # Define available operators.
    OPERATORS = {
        'max': (keras.layers.MaxPool1D, keras.layers.MaxPool2D, keras.layers.MaxPool3D),
        'avg': (keras.layers.AvgPool1D, keras.layers.AvgPool2D, keras.layers.AvgPool3D),
        'global_max': (keras.layers.GlobalMaxPool1D, keras.layers.GlobalMaxPool2D, keras.layers.GlobalMaxPool3D),
        'global_avg': (keras.layers.GlobalAvgPool1D, keras.layers.GlobalAvgPool2D, keras.layers.GlobalAvgPool3D)
    }
    # Define available arguments parsers.
    PARSERS = {
        'global': PARSER,
        'local': copy(PARSER).add('size', 'pool_size', 2).add('strides', None, None).add('padding', None, 'valid')
    }

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(Pooling, self).__init__(**kwargs)
        # Assign variables.
        self.method = str(self.args.pop('method', 'max')).lower()

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return super(Pooling, self).kwargparser(**kwargs).add('method', None, 'max')

    def build(self, input_shape):
        """
        Building process of layer when invoked.
        :param input_shape: shape of input.
        :return:            any.
        """
        # Calculate input dimension.
        dim = len(input_shape) - 2
        # Check method.
        methods = list(Pooling.OPERATORS.keys())
        assert self.method in methods, \
            self.message('method %s not found. Try again with %s.' % (self.method, generic.content(methods)))
        # Assign variables.
        ops = Pooling.OPERATORS[self.method]
        ndim = len(ops)
        # Check dimension.
        assert 0 < dim <= ndim, \
            self.message('%s pool only supports dimension %s' % (self.method, ['%sD' % (i + 1) for i in range(ndim)]))
        # Parse arguments.
        self.parser = Pooling.PARSERS['global'] if 'global' in self.method else Pooling.PARSERS['local']
        self.kwargs, self.args = self.parse(**self.kwargs)
        # Assign operator function.
        self.func = ops[dim - 1](**self.args)
