#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
import os
from tensorflow import keras
from brainium.nn import tensor
from brainium.nn.utils import Term
from brainium.common import KwargParse
from brainium.nn.utils import keras_model_visualize, keras_layer_visualize


class Model(keras.Model):
    """
    Groups neuron network layers into a model object with training and inference features.
    ---------
    @author:    Hieu Pham.
    @created:   16th March, 2020.
    @modified:  13rd August, 2020.
    """
    def __init__(self, inputs=None, outputs=None, **kwargs):
        """
        Model constructor.
        :param inputs:      inputs of model.
        :param outputs:     outputs of model.
        :param name:        proposed name of model.
        :param kwargs:      additional keyword arguments.
        """
        # Checking inputs.
        inputs = tensor.make(inputs)
        # Create begin and end cursors.
        begin, end = tensor.make(inputs, 0), tensor.make(inputs, -1)
        # Create temp model if outputs is exist.
        if outputs is not None:
            end = keras.Model(inputs=end, outputs=tensor.make(outputs, -1))
        # Make model taxonomy.
        term = Term(self, **kwargs)
        # Create custom model network.
        end = self.network(tensor.make(end, -1), **(self.parse(term=term, **kwargs)[-1]))
        # Create model by invoking super constructor.
        super(Model, self).__init__(inputs=begin, outputs=end, name=term.fullname)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return KwargParse()

    def parse(self, term, **kwargs) -> dict:
        """
        Parse arguments from input keyword arguments.
        :param kwargs:  keyword arguments to be parsed.
        :return:        arguments dict.
        """
        # Assign variables
        args = kwargs.copy()
        parser = self.kwargparser(**kwargs)
        # Get special config arguments.
        if '_conf' in kwargs:
            args.update(kwargs.get(term.basename, dict()))
            args.update(kwargs.get(term.fullname, dict()))
        # Return parsed arguments.
        return args, parser.parse(None, **args)

    def network(self, inputs=None, **kwargs):
        """
        Build model network.
        :param inputs:  inputs of model.
        :param kwargs:  additional keyword arguments.
        :return:        outputs of model.
        """
        # Let inherited models implement their own network.
        return inputs

    @staticmethod
    def connect(a=None, b=None, name='default'):
        """
        Connect two model into combined one.
        :param a:       first model.
        :param b:       second model.
        :param name:    proposed name of model.
        :return:        connected model.
        """
        # Check if both a and b are type of model.
        if isinstance(a, keras.Model) and isinstance(b, keras.Model):
            # Combine two model.
            return Model(inputs=a.input, outputs=b(a), name=name)
        # Otherwise, raise an error.
        else:
            raise ValueError('Both a and b must be model to be connected.')

    def trim(self, start=None, end=None, name=None):
        """
        Trim from start to end layers and pack them to new model.
        :param start:   start layer name or index.
        :param end:     end layer name or index.
        :param name:    proposed name of model.
        :return:        trimmed model.
        """
        # Get necessary tensor layers.
        a, b = tensor.make(self, index=start), tensor.make(self, index=end)
        # Create trimmed model.
        return Model(inputs=a, outputs=b, name=name)

    def plot(self, path=os.path.join(os.getcwd(), 'logs/models'), vertical=True, dpi=192):
        """
        Plot model into an image file.
        :param path:        path to image folder.
        :param vertical:    plot model in vertical or horizontal direction.
        :param dpi:         image quality.
        """
        # Checking directory.
        os.makedirs(path, exist_ok=True)
        # tensor.make file path.
        path = os.path.join(path, '%s.png' % self.name)
        # Plot an image to file.
        keras_model_visualize.plot_model(
            dpi=dpi,
            model=self,
            to_file=path,
            show_shapes=True,
            expand_nested=True,
            show_layer_names=True,
            rankdir='TB' if vertical else 'LR'
        )

    def summary(self, line_length=120, positions=None, print_fn=None):
        if not self.built:
            raise ValueError('This model has not yet been built. '
                             'Build the model first by calling `build()` or calling '
                             '`fit()` with some data, or specify '
                             'an `input_shape` argument in the first layer(s) for '
                             'automatic build.')
        keras_layer_visualize.print_summary(self, line_length=line_length, positions=positions, print_fn=print_fn)