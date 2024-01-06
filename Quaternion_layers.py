import numpy                    as np
from   numpy.random             import RandomState
import torch
from   torch.autograd           import Variable
import torch.nn.functional      as F
import torch.nn                 as nn
from   torch.nn.parameter       import Parameter
from   torch.nn.modules.utils   import _pair, _quadruple
from   torch.nn                 import Module
from    torch.distributions     import uniform
from   Quaternion_ops           import *
import math
import sys

class QuaternionTransposeConv(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.output_padding    = output_padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             = {'quaternion': quaternion_init,
                                  'unitary'   : unitary_init,
                                  'random'    : random_init}[self.weight_init]


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.out_channels, self.in_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_tranpose_conv_rotation(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilatation, self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) + ')'

class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilatation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilatation        = dilatation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilatation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'

class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=True, scale=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale    = scale

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel  = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, self.scale_param)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', rotation='       + str(self.rotation) \
            + ', seed=' + str(self.seed) + ')'

class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='he', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features  = in_features//4
        self.out_features = out_features//4
        self.r_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias     = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input  = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'

class QuaternionMaxAmpPool2d(Module):

    def __init__(self, kernel_size=2, stride=2, padding=0, same=False):
        super(QuaternionMaxAmpPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # print('size of input: ', x.shape)
        bs, c, h, w = x.shape
        # x = F.pad(x, self._padding(x), mode='constant')
        amp = get_modulus(x, vector_form=True)
        # print('size of amplitude: ', amp.shape)
        
        x_r = get_r(x)
        x_i = get_i(x)
        x_j = get_j(x)
        x_k = get_k(x)

        amp_uf = amp.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        # print("amp unfold size: ", amp_uf.size())
        x_r = x_r.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x_i = x_i.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x_j = x_j.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x_k = x_k.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        """
        tensor.unfold(axis, size, step)
        Returns a view of the original tensor which contains all slices of size size from self tensor in the dimension dimension.
        Step between two slices is given by step.
        """       
        amp_uf = amp_uf.contiguous().view(amp_uf.size()[:4] + (-1,))
        # print("amp size after view: ",amp_uf.size())
        x_r = x_r.contiguous().view(x_r.size()[:4] + (-1,))
        x_i = x_i.contiguous().view(x_i.size()[:4] + (-1,))
        x_j = x_j.contiguous().view(x_j.size()[:4] + (-1,))
        x_k = x_k.contiguous().view(x_k.size()[:4] + (-1,))
        """
        tensor.contiguous() => create a copy of tensor
        """
        maxamp = amp_uf.argmax(-1, keepdim=True)
        # print(maxamp.shape)
        """
        apply max amplitude position => max amplitude pooling
        """
        x_r_out = x_r.gather(-1, maxamp).view(bs, c//4, h // 2, w // 2)
        x_i_out = x_i.gather(-1, maxamp).view(bs, c//4, h // 2, w // 2)
        x_j_out = x_j.gather(-1, maxamp).view(bs, c//4, h // 2, w // 2)
        x_k_out = x_k.gather(-1, maxamp).view(bs, c//4, h // 2, w // 2)

        result = torch.cat((x_r_out, x_i_out, x_j_out, x_k_out), dim = 1)
        # print('size of pooling: ', result.size())

        return result
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'kernel_size=' + str(self.k) \
               + ', stride=' + str(self.stride) \
               + ', padding=' + str(self.padding) \
               + ', same=' + str(self.same) + ')'

class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, momentum = 0.9, training=True):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.momentum = momentum
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.training = training
        self.eps = torch.tensor(1e-5)

        self.register_buffer('running_mean_r', torch.zeros(self.num_features))
        self.register_buffer('running_mean_i', torch.zeros(self.num_features))
        self.register_buffer('running_mean_j', torch.zeros(self.num_features))
        self.register_buffer('running_mean_k', torch.zeros(self.num_features))
        self.register_buffer('running_variance', torch.ones(self.num_features))        
        """
        you use register_buffer when:
        you want a stateful part of your model that is not a parameter, but you want it in your state_dict
        registered buffers are Tensors (not Variables) 
        """
        self.running_mean_r.zero_()
        self.running_mean_i.zero_()
        self.running_mean_j.zero_()
        self.running_mean_k.zero_()
        self.running_variance.fill_(1)

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):

        quat_components = torch.chunk(input, 4, dim=1)
        n, c, h, w = input.shape
        r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]

        if self.training:
            mean_r = torch.mean(r,dim=(0,2,3))
            mean_i = torch.mean(i,dim=(0,2,3))
            mean_j = torch.mean(j,dim=(0,2,3))
            mean_k = torch.mean(k,dim=(0,2,3))
            # print(mean_r.shape)
            delta_r = r - mean_r.reshape((1, c//4, 1, 1))
            delta_i = i - mean_i.reshape((1, c//4, 1, 1))
            delta_j = j - mean_j.reshape((1, c//4, 1, 1))
            delta_k = k - mean_k.reshape((1, c//4, 1, 1))
            # print(delta_r.shape)
            quat_variance = torch.mean((delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2),dim=(0,2,3))
            
            with torch.no_grad():
                # print(self.running_mean_r.shape, mean_r.shape)
                self.running_mean_r = (self.momentum * self.running_mean_r) + (1.0-self.momentum) * mean_r
                self.running_mean_i = (self.momentum * self.running_mean_i) + (1.0-self.momentum) * mean_i
                self.running_mean_j = (self.momentum * self.running_mean_j) + (1.0-self.momentum) * mean_j
                self.running_mean_k = (self.momentum * self.running_mean_k) + (1.0-self.momentum) * mean_k
                self.running_variance = (self.momentum * self.running_variance) + (1.0-self.momentum) * (n*1.0 / (n - 1)*quat_variance)
        else:
            mean_r = self.running_mean_r
            mean_i = self.running_mean_i
            mean_j = self.running_mean_j
            mean_k = self.running_mean_k
            quat_variance = self.running_variance           
        # print(quat_variance.shape)
        denominator = torch.sqrt(quat_variance + self.eps).reshape((1, c//4, 1, 1))
        # print(denominator.shape)
        # print(r.shape)
        # Normalize
        r_normalized = (r - mean_r.reshape((1, c//4, 1, 1))) / denominator
        i_normalized = (i - mean_i.reshape((1, c//4, 1, 1))) / denominator
        j_normalized = (j - mean_j.reshape((1, c//4, 1, 1))) / denominator
        k_normalized = (k - mean_k.reshape((1, c//4, 1, 1))) / denominator

        beta_components = torch.chunk(self.beta, 4, dim=1)

        # Multiply gamma (stretch scale) and add beta (shift scale)
        new_r = (self.gamma * r_normalized) + beta_components[0]
        new_i = (self.gamma * i_normalized) + beta_components[1]
        new_j = (self.gamma * j_normalized) + beta_components[2]
        new_k = (self.gamma * k_normalized) + beta_components[3]

        new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)

        return new_input

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma) \
               + ', beta=' + str(self.beta) \
               + ', eps=' + str(self.eps) \
               + ', trainging_mode=' + str(self.training) + ')'             