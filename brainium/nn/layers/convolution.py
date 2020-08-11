# ----------------------------------------------------------------------------------------------------------------------
#  Copyright (c) 2020, The Brainium Authors. All rights reserved.                                                      -
#  ---------------------------------------------------------------------------------------------------------------------
#  Licensed under CC BY-NC-SA 4.0.                                                                                     -
#  You can use and adapt materials for non-commercial purposes as long as giving                                       -
#  appropriate credit by citing the authors. If you adapt the materials, you must                                      -
#  distribute your contributions under the same license as the original.                                               -
# ----------------------------------------------------------------------------------------------------------------------
from tensorflow import keras
from copy import deepcopy as copy
from brainium.nn.layers import Layer
from brainium.common import generic, KwargParse


# Define default keyword arguments parser of convolutions.
PARSER = KwargParse()
# Number of filters. We use 128 filters as the default.
PARSER.add('filters', None, 64)
# Convolution kernel size. We use 3x3 as the default.
PARSER.add('kernel', 'kernel_size', 3)
# Strides window size. We use 1x1 as the default.
PARSER.add('strides', None, 1)
# Convolution bias. We use bias as the default.
PARSER.add('bias', 'use_bias', True)
# Padding style of convolution. We use 'valid' as the default.
PARSER.add('padding', None, 'valid')
# Dilation size of convolution. We use 1x1 as the default.
PARSER.add('dilation', 'dilation_rate', 1)
# Regularization (weight decay) of convolution. We use L2=4e-5 follow Google's Xception paper.
# @reference: https://arxiv.org/abs/1610.02357.
PARSER.add('weight_decay', 'kernel_regularizer', keras.regularizers.l2(4e-5))
# Thin mode use depth-wise separable convolution which saves computation cost and reduces errors.
# @reference: https://arxiv.org/abs/1610.02357.
PARSER.add('thin', None, False)
# Data format. We follow format of the backend as the default.
PARSER.add('data_format', None, keras.backend.image_data_format())


class Convolution(Layer):
    """
    Common neuron network operation based on convolution matrix.
    ---------
    @author:    Hieu Pham.
    @created:   2nd August, 2020.
    """
    # Define available operators.
    OPERATORS = {
        'thin': [keras.layers.SeparableConv1D, keras.layers.SeparableConv2D],
        'none': [keras.layers.Conv1D, keras.layers.Conv2D, keras.layers.Conv3D]
    }

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        # Invoke super constructor.
        super(Convolution, self).__init__(**dict(kwargs, name='conv'))
        # Assign variables.
        self.thin = self.args.pop('thin', False)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        a keyword arguments parser.
        """
        return PARSER

    def build(self, input_shape):
        """
        Building process of layer when invoked.
        :param input_shape: shape of input.
        :return:            anything.
        """
        # Calculate input dimension.
        dim = len(input_shape) - 2
        # Assign variables.
        ops = Convolution.OPERATORS['thin'] if self.thin else Convolution.OPERATORS['none']
        max_dims = len(ops)
        # Check input dimension.
        if not 0 < dim <= max_dims:
            message = generic.content(['%sD' % i for i in range(1, max_dims + 1)])
            raise AssertionError(self.message('only supports dimension: %s' % message))
        # Assign operator function.
        self.func = ops[dim - 1](**self.args)

    def title(self) -> str:
        """
        Generate title of layer when summary.
        :return: title of layer.
        """
        return 'thin_' + self.name if self.thin else self.name

    def detail(self) -> str:
        """
        Generate the details of layer when plotted.
        :return: details of layer.
        """
        # Assign variables.
        args = self.args
        # Generate detail.
        detail = 'f:%s, k:%s, s:%s, %s' % (args['filters'], args['kernel_size'], args['strides'], args['padding'])
        if args['dilation_rate'] > 1:
            detail = '%s, d=%s' % (detail, args['dilation_rate'])
        detail = '%s.' % detail
        # Return detail.
        return detail


class SubPixelConvolution(Layer):
    """
    Alternative deconvolution for traditional transpose convolution which eliminates meaning less zero values
    anh use phase shift algorithm to generate more realistic image.
    @reference: https://arxiv.org/abs/1609.05158
    ---------
    @author:    Hieu Pham.
    @created:   6th August, 2020.
    """
    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs: keyword arguments to be passed.
        """
        # Invoke super constructor.
        super(SubPixelConvolution, self).__init__(**dict(kwargs, name='pixel_conv'))
        # Assign variables.
        self.r = self.args.pop('r', 2)
        self.thin = self.args.pop('thin', False)
        self.args.update(filters=int(self.r * self.r * self.args.pop('filters', 64)))

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return copy(PARSER).add('r', None, 2)

    def build(self, input_shape):
        """
        Building process of layer when invoked.
        :param input_shape: shape of input.
        :return:            anything.
        """
        # Calculate input dimension.
        dim = len(input_shape) - 2
        # Check dimension.
        assert dim == 2, self.message('only supports dimension 2D.')
        # Assign function operator.
        self.func = keras.layers.SeparableConv2D(**self.args) if self.thin else keras.layers.Conv2D(**self.args)

    def phase_shift(self, x):
        """
        Implementation of phase shift algorithm.
        :param x:   input tensor.
        :return:    shifted tensor.
        """
        # Assign variables.
        r = self.r
        batch, a, b, c = x.get_shape().as_list()
        # Handling Dimension(None) type for undefined batch dim
        batch = keras.backend.shape(x)[0]
        x = keras.backend.reshape(x, [batch, a, b, int(c / (r * r)), r, r])
        x = keras.backend.permute_dimensions(x, (0, 1, 2, 5, 4, 3))
        # Keras backend does not support tf.split, so we implement it.
        x = [x[:, i, :, :, :, :] for i in range(a)]
        x = keras.backend.concatenate(x, 2)
        x = [x[:, i, :, :, :] for i in range(b)]
        x = keras.backend.concatenate(x, 2)
        # Return shifted tensor.
        return x

    def call(self, inputs, **kwargs):
        """
        Apply processing steps.
        :param inputs:  input tensor.
        :param kwargs:  additional keyword arguments to be passed.
        :return:        output tensor.
        """
        return self.phase_shift(self.func(inputs, **kwargs))

    def compute_output_shape(self, input_shape):
        """
        Compute output shape of the layer.
        :param input_shape: shape of input tensor.
        :return:            shape of output tensor.
        """
        shape = self.func.compute_output_shape(input_shape)
        shape = (shape[0], self.r * shape[1], self.r * shape[2], shape[3] / (self.r * self.r))
        return shape

    def get_config(self):
        """
        Generate configuration of the layer.
        :return: config dict.
        """
        config = self.func.get_config()
        config.pop('rank')
        config.pop('dilation_rate')
        config['filters'] /= self.r * self.r
        config['r'] = self.r
        return config


class TransposeConvolution(Layer):
    """
    Upsample the input feature map to a desired output feature map using some learnable parameters.
    ---------
    @author:    Hieu Pham.
    @created:   11st August, 2020.
    """
    # Define available operators.
    OPERATORS = (keras.layers.Conv1DTranspose, keras.layers.Conv2DTranspose, keras.layers.Conv3DTranspose)

    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        super(TransposeConvolution, self).__init__(**kwargs)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return PARSER

    def build(self, input_shape):
        """
        Building process of layer when invoked.
        :param input_shape: shape of input.
        :return:            anything.
        """
        # Calculate input dimension.
        dim = len(input_shape) - 2
        # Calculate max dimension.
        mdim = len(TransposeConvolution.OPERATORS)
        # Check dimension.
        assert 0 < dim <= mdim, \
            self.message('only supports dimension %s' % generic.content(['%sD' % (i + 1) for i in range(mdim)]))
        # Assign operator function.
        self.func = TransposeConvolution.OPERATORS[dim](**self.args)


class Deconvolution(Layer):
    """
    The process of filtering a signal to compensate for an undesired convolution.
    ---------
    @author:    Hieu Pham.
    @created:   6th August, 2020.
    @modified:  11st August, 2020.
    """
    def __init__(self, **kwargs):
        """
        Class constructor.
        :param kwargs:  keyword arguments to be passed.
        """
        # Invoke super constructor.
        super(Deconvolution, self).__init__(**dict(kwargs, name='deconv'))
        # Assign variables.
        self.method = str(self.args.pop('method', None)).lower()
        # Check method.
        assert self.method in ('pixel', 'none'), \
            self.message('method %s not found. Try again with pixel or none.' % self.method)
        # Assign operator function.
        self.func = SubPixelConvolution(**self.kwargs) if self.method == 'pixel' else TransposeConvolution(**self.args)

    def kwargparser(self, **kwargs) -> KwargParse:
        """
        Get keyword arguments parser.
        :param kwargs:  keyword arguments to be passed.
        :return:        keyword arguments parser.
        """
        return KwargParse().add('method', None, None)

    def get_config(self):
        """
        Generate configuration of the layer.
        :return: config dict.
        """
        return self.func.get_config()

    def title(self) -> str:
        """
        Generate title of layer when summary.
        :return: title of layer.
        """
        return '%s_%s' % (self.method, self.name) if self.method != 'none' else self.name

    def detail(self) -> str:
        """
        Generate the details of layer when plotted.
        :return: details of layer.
        """
        # Assign variables.
        args = self.args
        # Generate detail.
        detail = 'f:%s, k:%s, s:%s, %s' % (args['filters'], args['kernel_size'], args['strides'], args['padding'])
        if args['dilation_rate'] > 1:
            detail = '%s, d=%s' % (detail, args['dilation_rate'])
        if self.method == 'pixel':
            detail = '%s, r=%s' % (detail, self.func.r)
        detail = '%s.' % detail
        # Return detail.
        return detail