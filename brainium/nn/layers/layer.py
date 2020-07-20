# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from abc import ABC
from types import LambdaType
from tensorflow.keras import layers
from brainium.nn.utils import Taxonomy
from brainium.common import KwargParser


# ----------------------------------------------------------------------------------------------------------------------
# NEURON NETWORK LAYER
# A layer is an implementation of a neuron network operation.
# ----------------------------------------------------------------------------------------------------------------------
class Layer(layers.Layer, ABC):
    """
    This class is base class of all inherited neuron network layers.
    ---------
    @author:    Hieu Tr. Pham.
    @modified:  20th July, 2020.
    """

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        # Init default operation.
        self.operation = None
        # Init layer taxonomy.
        self.taxonomy = Taxonomy(self, **kwargs)
        # Init layer arguments.
        self.args = self.parse_args(**kwargs)
        # Invoke parent constructor.
        super(Layer, self).__init__(name=self.taxonomy.fullname)

    def schema(self, **kwargs) -> dict:
        """
        Get arguments schema of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments schema.
        """
        return dict()

    def keymap(self, **kwargs) -> dict:
        """
        Get arguments keymap of layer.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        arguments keymap.
        """
        return dict()

    def parse_args(self, **kwargs) -> dict:
        """
        Parse layer arguments follow a special way that allows layer to be flexible configured.
        :param kwargs:  keyword arguments to be parsed.
        :return:        dict that describes arguments.
        """
        # Make default arguments.
        args = KwargParser(self.schema(**kwargs)).parse(nested=True)
        # Parse arguments from config.
        config = kwargs.get('_config', dict())
        args = KwargParser(args).parse(nested=True, **config.get(self.taxonomy.basename, dict()))
        args = KwargParser(args).parse(nested=True, **config.get(self.taxonomy.proposal, dict()))
        args = KwargParser(args).parse(nested=True, **config)
        args = KwargParser(args, self.keymap(**kwargs)).parse(nested=True, **kwargs)
        # Return final arguments.
        return args

    def call(self, inputs, **kwargs):
        """
        Apply processing steps.
        :param inputs:  inputs tensor.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        processed tensor.
        """
        # Verify layer operation.
        assert isinstance(self.operation, (layers.Layer, LambdaType)), \
            self.message('neuron network operation must be layer or lambda function.')
        # Return tensor processed by operation.
        return self.operation(inputs, **kwargs)

    def title(self) -> str:
        """
        Generate title of layer when summary.
        :return: title of layer.
        """
        return self.name

    def detail(self) -> str:
        """
        Generate the details of layer when plotted.
        :return: details of layer.
        """
        return self.name

    def message(self, message):
        """
        Generate message which includes class name.
        :param message: string of message.
        :return:        string of message included class name.
        """
        return '%s: %s' % (self.__class__.__name__, message)