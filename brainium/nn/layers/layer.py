# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from abc import ABC
from tensorflow import keras
from brainium.nn.utils import Term
from brainium.common import KwargParse, Common


class Layer(keras.layers.Layer, Common, ABC):
    """
    Basic neuron network layer which is base class of others inherited.
    ---------
    @author:    Hieu Pham.
    @created:   28th July, 2020.
    """
    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        # Assign variables.
        self.func = None
        self.term = Term(self, **kwargs)
        self.parser = self.kwargparse(**kwargs)
        self.args = self.parse(**kwargs)
        # Invoke parent constructor.
        super(Layer, self).__init__(name=self.term.fullname)

    def kwargparse(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        a keyword arguments parser.
        """
        return KwargParse()

    def parse(self, **kwargs) -> dict:
        """
        Parse layer arguments from input keyword arguments.
        :param kwargs:  keyword arguments to be passed.
        :return:        arguments dict.
        """
        args = kwargs.copy()
        # Get special config arguments.
        if '_conf' in kwargs:
            args.update(kwargs.get(self.term.basename, dict()))
            args.update(kwargs.get(self.term.fullname, dict()))
        # Return parsed arguments.
        return self.parser.parse(None, **args)

    def call(self, inputs, **kwargs):
        """
        Apply processing steps.
        :param inputs:  inputs tensor.
        :param kwargs:  keyword arguments to be passed.
        :return:        processed tensor.
        """
        assert callable(self.func), self.message('function must be callable.')
        return self.func(inputs, **kwargs)

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