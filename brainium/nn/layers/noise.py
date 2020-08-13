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
from brainium.common import KwargParse, generic
from tensorflow.keras.layers import GaussianNoise


class SpatialDropout(Layer):
    """
    Variant version of dropout that drops entire 2D feature maps instead of individual elements.
    ---------
    @author:    Hieu Pham.
    @created:   12nd August, 2020.
    """
    # Define available operators.
    OPERATORS = (keras.layers.SpatialDropout1D, keras.layers.SpatialDropout2D, keras.layers.SpatialDropout3D)

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(SpatialDropout, self).__init__(**kwargs)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return KwargParse().add('rate', None, 0.2).add('data_format', None, keras.backend.image_data_format())

    def build(self, input_shape):
        """
        Building process of layer when invoked.
        :param input_shape: shape of input.
        :return:            any.
        """
        # Calculate input dimension.
        dim = len(input_shape) - 2
        # Assign operators.
        ops = SpatialDropout.OPERATORS
        # Check dimension.
        assert 0 < dim <= len(ops), \
            self.message('only supports dimension %s.' % generic.content(['%sD' % (i + 1) for i in range(len(ops))]))
        # In case 1D, remove data format from arguments.
        if dim == 1:
            self.args.pop('data_format', None)
        # Assign operator function.
        self.func = ops[dim - 1](**self.args)


class Dropout(Layer):
    """
    Random drop units from neural network during training.
    ---------
    @author:    Hieu Pham.
    @created:   12nd August, 2020.
    """
    # Define available operators.
    OPERATORS = {
        'none': {
            'func': keras.layers.Dropout,
            'parser': KwargParse().add('rate', None, 0.2).add('shape', 'noise_shape', None).add('seed', None, None)
        },
        'alpha': {
            'func': keras.layers.AlphaDropout,
            'parser': KwargParse().add('rate', None, 0.2).add('shape', 'noise_shape', None).add('seed', None, None)
        },
        'guassian': {
            'func': keras.layers.GaussianDropout,
            'parser': KwargParse().add('rate', None, 0.2)
        },
        'spatial': {
            'func': SpatialDropout
        }
    }

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(Dropout, self).__init__(**kwargs)
        # Assign method.
        self.method = str(self.args.pop('method', None)).lower()
        # Assign variables.
        ops = Dropout.OPERATORS
        methods = list(ops.keys())
        # Check method.
        assert self.method in methods, \
            self.message('method %s not found. Try again with %s.' % (self.method, methods))
        # Assign operator function.
        ops = ops[self.method]
        self.func = ops['func'](**(ops['parser'].parse(**self.kwargs) if 'parser' in ops else self.kwargs))