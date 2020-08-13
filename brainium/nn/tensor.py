#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
import tensorflow as tf


def make(inputs, index=None):
    """
    Make tensor based on inputs.
    :param inputs:  inputs object.
    :param index:   index of layer would be taken.
    :return:        output tensor(s).
    ---------
    @author:    Hieu Pham.
    @created:   19th March, 2020.
    """
    # In case inputs is None, return default input layer 128 x 128 x 3.
    if inputs is None:
        return tf.keras.Input((128, 128, 3))
    # In case inputs is a Tensor, return itself.
    elif isinstance(inputs, tf.Tensor):
        return inputs
    # In case inputs is a shape, return input layer within that shape.
    elif isinstance(inputs, tuple):
        return tf.keras.Input(inputs)
    # In case inputs is a model.
    elif isinstance(inputs, tf.keras.Model):
        assert index is None or isinstance(index, (int, str)), 'Tensor index should be integer or string.'
        return inputs if index is None \
            else inputs.get_layer(index=index) if isinstance(index, int) \
            else inputs.get_layer(name=index)
    # In case inputs is a layer.
    elif isinstance(inputs, tf.keras.layers.Layer):
        return inputs.input if index else inputs.output
    # Otherwise, raise an error.
    raise ValueError('Tensor can only take from shape(s), layer(s) or model(s).')