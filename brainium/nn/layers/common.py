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
from brainium.common import KwargParse


class Flattens(Layer):
    """
    Flatten input tensor into 1D vector.
    ---------
    @author:    Hieu Pham.
    @created:   12nd August, 2020.
    """
    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(Flattens, self).__init__(**kwargs)
        # Assign operator function.
        self.func = keras.layers.Flatten(**self.args)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        a keyword arguments parser.
        """
        return super(Flattens, self).kwargparser(**kwargs)


class Denses(Layer):
    """
    Connect each input neuron to all neurons of this layer.
    ---------
    @author:    Hieu Pham.
    @created:   12nd August, 2020.
    """
    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(Denses, self).__init__(**kwargs)
        # Assign operator function.
        self.func = keras.layers.Dense(**self.args)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        a keyword arguments parser.
        """
        return KwargParse().add('units', None, 128)
