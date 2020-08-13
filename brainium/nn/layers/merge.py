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


class Merge(Layer):
    """
    ---------
    @author:    Hieu Pham.
    @created:   13rd August, 2020.
    """
    # Define available operators.
    OPERATORS = {
        'add': {'func': keras.layers.Add},
        'sub': {'func': keras.layers.Subtract},
        'mul': {'func': keras.layers.Multiply},
        'min': {'func': keras.layers.Minimum},
        'max': {'func': keras.layers.Maximum},
        'concat': {
            'func': keras.layers.Concatenate,
            'parser': KwargParse().add('axis', None, -1)
        },
        'dot': {
            'func': keras.layers.Dot,
            'parser': KwargParse().add('axes', None, -1).add('normalize', None, False)
        }
    }

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(Merge, self).__init__(**kwargs)
        # Assign method.
        self.method = str(self.args.pop('method', 'add')).lower()
        # Assign operators.
        ops = Merge.OPERATORS
        # Check method.
        assert self.method in ops, \
            self.message('method %s not found. Try again with %s.' % (self.method, generic.content(list(ops.keys()))))
        # Assign operator function.
        ops = ops[self.method]
        self.func = ops['func'](**(ops['parser'].parse(**self.kwargs) if 'parser' in ops else self.kwargs))

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return KwargParse().add('method', None, 'add')

    def title(self) -> str:
        """
        Generate title of layer when summarized.
        :return: title of layer.
        """
        return '%s_merge_%s' % (self.method, self.term.suffix)

    def detail(self) -> str:
        """
        Generate details of layer when plotted.
        :return: details of layer.
        """
        return 'method: %s' % self.method