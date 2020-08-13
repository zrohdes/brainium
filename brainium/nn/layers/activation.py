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
from tensorflow.keras.regularizers import l2


# Define supported activations and their kwarg parsers.
ACTIVATIONS = {
    # ReLU (Rectified Linear Unit).
    # A piecewise linear function that will output the input directly if is positive, otherwise, it will output zero.
    'relu': {
        'func': keras.layers.ReLU,
        'parse': KwargParse().add('max', 'max_value', 0).add('slope', 'negative_slop', 0).add('threshold', None, 0)
    },
    # Leaky ReLU (Leaky Rectified Linear Unit).
    # A variant of ReLU that allows small negative values when the input is less than zero.
    'leaky': {
        'func': keras.layers.LeakyReLU,
        'parse': KwargParse().add('alpha', None, 0.02)
    },
    # PReLU (Parametric Rectified Linear Unit).
    # A variant of LeakyReLU with trainable alpha coefficient.
    'prelu': {
        'func': keras.layers.PReLU,
        'parse': KwargParse().add('weight_decay', 'alpha_regularizer', l2(5e-6)).add('axes', 'shared_axes', None)
    },
    # ThresholdReLU (Threshold Rectified Linear Unit)
    # A variant of ReLU with selectable threshold.
    'threshold': {
        'func': keras.layers.ThresholdedReLU,
        'parse': KwargParse().add('theta', None, 0.1)
    },
    # Sigmoid.
    # Logistic function that limits the output to a range between 0 and 1.
    'sigmoid': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'sigmoid')
    },
    # Hard sigmoid.
    # A variant of sigmoid which faster and cheaper computation than original version.
    'hard_sigmoid': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'hard_sigmoid')
    },
    # Softmax.
    # Turn logits into probabilities that sum to one by take the exponents of each output.
    'softmax': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'softmax')
    },
    # Tanh.
    # A scaled version of sigmoid demonstrated by tanh function that limits output from -1 to 1.
    'tanh': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'tanh')
    },
    # Softsign.
    # A variant of tanh which converge by polynomial.
    'softsign': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'softsign'),
    },
    # Softplus.
    # A variant of sigmoid which limit from zero to infinity.
    'softplus': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'softplus')
    },
    # ELU (Exponential Linear Unit).
    # Fixes some of the problems with ReLUs and keeps some of the positive things by exponent function.
    'elu': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'elu')
    },
    # SeLU (Scaled Exponential Linear Unit).
    # A variant of ELU with scaled lambda and alpha coefficient.
    'selu': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'selu')
    },
    # Linear.
    'linear': {
        'func': keras.layers.Activation,
        'parse': KwargParse().add('method', 'activation', 'linear')
    }
}


class Activation(Layer):
    """
    Define outputs from a set of inputs based on an activation operation.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    @modified:  11st August, 2020.
    """
    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        super(Activation, self).__init__(**kwargs)
        # Assign method.
        self.method = str(self.args.get('method', 'sigmoid')).lower()
        # Checking method.
        methods = list(ACTIVATIONS.keys())
        assert self.method in methods, \
            self.message('method %s not found. Try again with: %s' % (self.method, generic.content(methods, end='or')))
        # Override parser for corrected activation method.
        self.parser = ACTIVATIONS[self.method]['parse']
        # Override arguments.
        self.kwargs, self.args = self.parse(**kwargs)
        # Assign activation function.
        self.func = ACTIVATIONS[self.method]['func'](**self.args)

    @staticmethod
    def available_methods():
        """
        Get available activation methods.
        :return: list of methods.
        """
        return list(ACTIVATIONS.keys())

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return super(Activation, self).kwargparser(**kwargs).add('method', None, 'sigmoid')

    def title(self) -> str:
        """
        Generate title of layer when summarized.
        :return: title of layer.
        """
        return '%s_%s' % (self.method, self.term.suffix)

    def detail(self) -> str:
        """
        Generate details of layer when plotted.
        :return: details of layer.
        """
        return 'method: %s' % self.method
