# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from tensorflow import keras
from brainium.common import generic
from brainium.nn.layers import Layer


# ----------------------------------------------------------------------------------------------------------------------
# ReLU (Rectified Linear Unit)
# A piecewise linear function that return the input directly if is positive, otherwise zeros.
# ----------------------------------------------------------------------------------------------------------------------
class ReLU(Layer):
    """
    A piecewise linear function that return the input directly if is positive, otherwise zeros.
    ---------
    @author:    Hieu Tr. Pham.
    @modified:  20th July, 2020.
    """

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        # Invoke parent constructor.
        super(ReLU, self).__init__(**kwargs)
        # Init ReLU operation.
        self.operation = keras.layers.ReLU(**self.args)

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs: additional keyword arguments to be passed.
        :return:       arguments schema.
        """
        return dict(max=None, slope=0, threshold=0)

    def keymap(self, **kwargs) -> dict:
        """
        Get arguments keymap of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments keymap.
        """
        return dict(max='max_value', slope='negative_slope')


# ----------------------------------------------------------------------------------------------------------------------
# Leaky ReLU (Leaky Rectified Linear Unit)
# A variant of ReLU that allows small negative values when the input is less than zero.
# ----------------------------------------------------------------------------------------------------------------------
class LeakyReLU(Layer):
    """
    A variant of ReLU that allows small negative values when the input is less than zero.
    ---------
    @author:    Hieu Tr. Pham.
    @modified:  20th July, 2020.
    """

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        # Invoke parent constructor.
        super(LeakyReLU, self).__init__(**kwargs)
        # Init LeakyReLU operation.
        self.operation = keras.layers.LeakyReLU(**self.args)

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments schema.
        """
        return dict(alpha=0.02)


# ----------------------------------------------------------------------------------------------------------------------
# PReLU (Parametric Rectified Linear Unit)
# A variant of LeakyReLU with trainable co-efficients.
# ----------------------------------------------------------------------------------------------------------------------
class PReLU(Layer):
    """
    A variant of LeakyReLU with trainable co-efficients.
    ---------
    @author:    Hieu Tr. Pham.
    @modified:  20th July, 2020.
    """

    def __init__(self, **kwargs):
        """
        Class constructor
        :param kwargs:  keyword arguments to be passed.
        """
        # Invoke parent constructor.
        super(PReLU, self).__init__(**kwargs)
        # Init PReLU operation.
        self.operation = keras.layers.PReLU(**self.args)

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments schema.
        """
        return dict(weight_decay=keras.regularizers.l2(5e-6), axes=None)

    def keymap(self, **kwargs) -> dict:
        """
        Get arguments keymap of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments keymap.
        """
        return dict(weight_decay='alpha_regularizer', axes='shared_axes')


# ----------------------------------------------------------------------------------------------------------------------
# Threshold Rectified Linear Unit (ThresholdReLU)
# A variant of ReLU with selectable threshold.
# ----------------------------------------------------------------------------------------------------------------------
class ThresholdReLU(Layer):
    """
    A variant of ReLU with selectable threshold.
    ---------
    @author:    Hieu Tr. Pham.
    @created:   12nd July, 2020.
    """

    def __init__(self, **kwargs):
        """
        Class construtcor.
        :param kwargs: keyword arguments to be passed.
        """
        # Invoke parent constructor.
        super(ThresholdReLU, self).__init__(**kwargs)
        # Init ThresholdReLU operation.
        self.operation = keras.layers.ThresholdedReLU(**self.args)

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments schema.
        """
        return dict(theta=0.1)


# ----------------------------------------------------------------------------------------------------------------------
# Activation
# Define outputs from a set of inputs based on an activation operation.
# ----------------------------------------------------------------------------------------------------------------------
class Activation(Layer):
    """
    Define outputs from a set of inputs based on an activation operation.
    ---------
    @author:    Hieu Tr. Pham.
    @modified:  20th July, 2020.
    """

    # Define available activation operations.
    OPERATIONS = {
        'relu': ReLU,
        'prelu': PReLU,
        'leaky': LeakyReLU,
        'threshold': ThresholdReLU,
        'elu,selu,sigmoid,hard_sigmoid,softmax,softsign,softplus,tanh,linear': keras.layers.Activation
    }

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        # Invoke parent constructor.
        super(Activation, self).__init__(**kwargs)
        # Get method.
        self.method = self.args.pop('method', 'sigmoid')
        # Get available methods.
        methods = Activation.get_available_methods()
        # Verify method.
        assert self.method in methods, self.message('only supports: %s' % generic.content(methods))
        # Init activation operation based on method.
        methods = list(Activation.OPERATIONS.keys())
        self.operation = (Activation.OPERATIONS[self.method])(**kwargs) if self.method in methods \
            else (Activation.OPERATIONS[methods[-1]])(activation=self.method)

    @staticmethod
    def get_available_methods():
        """
        Get available activation methods.
        :return: list of available activation methods.
        """
        methods = list(Activation.OPERATIONS.keys())
        methods.extend(methods.pop().split(','))
        return methods

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments schema.
        """
        return dict(method='sigmoid')

    def title(self) -> str:
        """
        Generate title of layer when summary.
        :return: title of layer.
        """
        return '%s_%s' % (self.method, self.taxonomy.suffix)

    def detail(self) -> str:
        """
        Generate the details of layer when plotted.
        :return: details of layer.
        """
        return 'act: %s' % self.method