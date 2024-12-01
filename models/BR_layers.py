from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
# from icecream import ic
import torch.nn.functional as F
import torch.nn as nn


# Three-hidden-layer neural network
class NN3(torch.nn.Module):
    def __init__(self, n_dim, n_width=32, n_out=None):
        super(NN3, self).__init__()
        self.n_dim = n_dim 
        self.n_width = n_width
        self.n_out = n_out

        self.l_s = torch.nn.Linear(n_dim, n_width)
        self.l_1 = torch.nn.Linear(n_width, n_width)
        self.l_2 = torch.nn.Linear(n_width, n_width)
        self.l_3 = torch.nn.Linear(n_width, n_width)
        self.l_f = torch.nn.Linear(n_width, n_out)

        nn.init.xavier_normal_(self.l_s.weight.data)
        nn.init.zeros_(self.l_s.bias.data)

        nn.init.xavier_normal_(self.l_1.weight.data)
        nn.init.zeros_(self.l_1.bias.data)

        nn.init.xavier_normal_(self.l_2.weight.data)
        nn.init.zeros_(self.l_2.bias.data)

        nn.init.xavier_normal_(self.l_3.weight.data)
        nn.init.zeros_(self.l_3.bias.data)

        nn.init.xavier_normal_(self.l_f.weight.data)
        nn.init.zeros_(self.l_f.bias.data)

    def forward(self, inputs):
        # relu with low regularity
        x = F.relu(self.l_s(inputs))
        x = F.relu(self.l_1(x))
        x = F.relu(self.l_2(x))
        x = F.relu(self.l_3(x))

        # tanh with high regularity
        #x = tf.nn.tanh(self.l_1(inputs))
        #x = tf.nn.tanh(self.l_2(x))
        #x = tf.nn.relu(self.l_3(x))
        #x = tf.nn.relu(self.l_4(x))

        x = self.l_f(x)

        return x
    
class NNx(torch.nn.Module):
    def __init__(self, layers):
        super(NNx, self).__init__()
        self.layers = layers
        self.n_depth = len(layers) - 2
        # self.iter = 0
        self.activation = nn.ReLU()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.actnorm_layer = nn.ModuleList([actnorm(layers[i+1]) for i in range(len(layers) - 2)])

        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, inputs):
        if not torch.is_tensor(inputs):
            x = torch.from_numpy(inputs)
        else:
            x = inputs

        
        x = self.linear[0](x)

        for i in range(1, len(self.layers)-2):
            y = x
            y = self.actnorm_layer[i-1](y)
            y = self.activation(y)
            y = self.linear[i](y)
            x = x + y

        x = self.actnorm_layer[-1](x)
        x = self.activation(x)
        x = self.linear[-1](x)

        return x
    
    def actnorm_data_initialization(self):
        for i in range(self.n_depth):
            self.actnorm_layer[i].reset_data_initialization()


class VAE_encoder(torch.nn.Module):
    def __init__(self, n_dim,
                 n_out_dim,   # number of dimensions
                 n_depth, # number of hidden layers.
                 n_width):
        super(VAE_encoder, self).__init__()

        self.n_dim = n_dim
        self.n_depth = n_depth
        self.n_width = n_width
        self.n_out = 2*n_out_dim
        self.n_out_dim = n_out_dim

        self.layers = [self.n_dim] + self.n_depth * [self.n_width] + [self.n_out]

        self.encoder = NNx(self.layers)

    def forward(self, inputs):
        y = inputs
        x = self.encoder(y)
        mean = x[:,:self.n_out_dim]
        std = torch.exp(x[:, self.n_out_dim:])+1.0e-6

        return mean, std

    def actnorm_data_initialization(self):
        self.encoder.actnorm_data_initialization()


# VAE Gaussian decoder. The form is the same as the encoder.
class VAE_decoder(torch.nn.Module):
    def __init__(self, n_dim,
                 n_out_dim,   # number of dimensions
                 n_depth, # number of hidden layers.
                 n_width):
        super(VAE_decoder, self).__init__()

        self.n_dim = n_dim
        self.n_depth = n_depth
        self.n_width = n_width
        self.n_out = 2*n_out_dim
        self.n_out_dim = n_out_dim

        self.layers = [self.n_dim] + self.n_depth * [self.n_width] + [self.n_out]


        self.decoder = NNx(self.layers)

        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        x = inputs
        y = self.decoder(x)
        mean = y[:,:self.n_out_dim]
        #std = tf.exp(y[:, self.n_dim:]) + 1.0e-6
        std = torch.exp(self.alpha)*(1.0+0.99*torch.tanh(y[:,self.n_out_dim:]))

        return mean, std

    def actnorm_data_initialization(self):
        self.decoder.actnorm_data_initialization()



class affine_coupling(torch.nn.Module):
    def __init__(self, n_dim, n_split_at, n_width=32, flow_coupling=1):
        super(affine_coupling, self).__init__()
        self.n_dim = n_dim
        self.n_split_at = n_split_at
        self.n_width = n_width
        self.flow_coupling = flow_coupling

        if self.flow_coupling == 0:
            self.f = NN3(n_split_at, n_width, n_dim-n_split_at)
        elif self.flow_coupling == 1:
            self.f = NN3(n_split_at, n_width, (n_dim-n_split_at)*2)
        else:
            raise Exception()
        self.log_gamma = torch.nn.Parameter(torch.zeros(1, n_dim-n_split_at))

    def forward(self, inputs, logdet=None, reverse=False):
        z = inputs
        n_split_at = self.n_split_at

        alpha = 0.6

        if not reverse:
            z1 = z[:,:n_split_at]
            z2 = z[:,n_split_at:]

            h = self.f(z1)
            shift = h[:,0::2]

            scale = alpha*F.tanh(h[:,1::2])
            #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            shift = torch.exp(torch.clamp(self.log_gamma, -5.0, 5.0))*F.tanh(shift)
            z2 = z2 + scale * z2 + shift
            if logdet is not None:
                dlogdet = torch.sum(torch.log(scale + torch.ones_like(scale)),
                                        dim=[1], keepdim=True)
                
            z = torch.cat((z1,z2), 1)
            
        else:
            z1 = z[:,:n_split_at]
            z2 = z[:,n_split_at:]

            h = self.f(z1)
            shift = h[:,0::2]

            # resnet-like trick
            # we suppressed both the scale and the shift.
            scale = alpha*F.tanh(h[:,1::2])
            #shift = tf.exp(self.log_gamma)*tf.nn.tanh(shift)
            shift = torch.exp(torch.clamp(self.log_gamma, -5.0, 5.0))*F.tanh(shift)
            z2 = (z2 - shift) / (torch.ones_like(scale) + scale)
            if logdet is not None:
                dlogdet = - torch.sum(torch.log(scale + torch.ones_like(scale)),
                                        dim=[1], keepdim=True)

            z = torch.cat((z1,z2), 1)

        if logdet is not None:
           return z, logdet + dlogdet
        
        return z 


class actnorm(torch.nn.Module):
    def __init__(self, n_dim, scale = 1.0, logscale_factor=3.0):
        super(actnorm, self).__init__()
        self.n_dim = n_dim 
        self.scale = scale
        self.logscale_factor = logscale_factor

        self.data_init = False

        self.b = torch.nn.Parameter(torch.zeros(1, n_dim))
        self.logs = torch.nn.Parameter(torch.zeros(1, n_dim))

        self.register_buffer('b_init', torch.zeros(1,n_dim))
        self.register_buffer('logs_init', torch.zeros(1,n_dim))

    def forward(self, inputs, logdet=None, reverse=False):
        assert inputs.shape[-1] == self.n_dim 

        if not self.data_init and not reverse:
            x_mean = torch.mean(inputs, dim=[0], keepdim=True)
            x_var = torch.mean(torch.square(inputs-x_mean), dim=[0], keepdim=True)

            self.b_init = - x_mean 
            self.logs_init = torch.log(self.scale/(torch.sqrt(x_var)+1e-6))/self.logscale_factor

            self.b_init = self.b_init.detach().clone()
            self.logs_init = self.logs_init.detach().clone() 

            self.data_init = True 

        if not reverse:
            x = inputs + (self.b + self.b_init)
            #x = x * tf.exp(self.logs + self.logs_init)
            x = x * torch.exp(torch.clamp(self.logs + self.logs_init, -5., 5.))
        else:
            #x = inputs * tf.exp(-self.logs - self.logs_init)
            x = inputs * torch.exp(-torch.clamp(self.logs + self.logs_init, -5., 5.))
            x = x - (self.b + self.b_init)

        if logdet is not None:
            #dlogdet = tf.reduce_sum(self.logs + self.logs_init)
            dlogdet = torch.sum(torch.clamp(self.logs + self.logs_init, -5., 5.))
            if reverse:
                dlogdet *= -1
            return x, logdet + dlogdet

        return x
    
    def reset_data_initialization(self):
        self.data_init = False 


class flow_mapping(torch.nn.Module):
    def __init__(self, n_dim, n_depth, n_split_at, n_width=32, n_bins=32, **kwargs):
        super(flow_mapping, self).__init__()
        self.n_dim = n_dim 
        self.n_depth = n_depth 
        self.n_split_at = n_split_at
        self.n_width = n_width 
        self.n_bins = n_bins 

        assert n_depth % 2 == 0

        self.scale_layers = torch.nn.ModuleList()
        self.affine_layers = torch.nn.ModuleList() 

        sign = -1
        for i in range(self.n_depth):
            self.scale_layers.append(actnorm(n_dim)) 
            sign *= -1
            i_split_at = (self.n_split_at*sign + self.n_dim) % self.n_dim
            self.affine_layers.append(affine_coupling(n_dim, 
                                                      i_split_at,
                                                      n_width=self.n_width))
    
    def forward(self, inputs, logdet=None, reverse=False):
        assert inputs.shape[-1] == self.n_dim 

        if not reverse:
            z = inputs 
            for i in range(self.n_depth):
                z = self.scale_layers[i](z, logdet)
                if logdet is not None:
                    z, logdet = z 

                z = self.affine_layers[i](z, logdet)
                if logdet is not None:
                    z, logdet = z 

                z = torch.flip(z, [1])

        else:
            z = inputs 

            for i in reversed(range(self.n_depth)):
                z = torch.flip(z, [1])

                z = self.affine_layers[i](z, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z 

                z = self.scale_layers[i](z, logdet, reverse=True)
                if logdet is not None:
                    z, logdet = z 

        if logdet is not None:
            return z, logdet 
        return z 
    
    def actnorm_data_initialization(self):
        for i in range(self.n_depth):
            self.scale_layers[i].reset_data_initialization()


class W_LU(nn.Module):
    """ KRnet WLU layer 
    act as a rotation layer, which rotate vars in different dimensions 
    """
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.LU = nn.Parameter(torch.zeros(input_size, input_size))

        self.register_buffer('LU_init', torch.eye(input_size, input_size))
        self.register_buffer('ones_mat', torch.ones(input_size, input_size))
        self.register_buffer('I',torch.eye(self.input_size))

    def forward(self, x, logdet=None, reverse=False):
        if not reverse:
            # invP*L*U*x
            LU = self.LU_init + self.LU

            # upper-triangular matrix
            U = torch.triu(LU) 

            # diagonal line
            U_diag = torch.diagonal(U) 

            # trainable mask for U
            U_mask = torch.triu(self.ones_mat)
            # U_mask = (torch.triu(self.ones_mat) >= 1)
            U = ((1-U_mask)*U).detach() + U_mask*U 

            # lower-triangular matrix
            
            L = torch.tril(self.I+LU)-torch.diagonal(LU).diag()

            # trainable mask for L
            L_mask = torch.tril(self.ones_mat) - self.I

            L = ((1-L_mask)*L).detach() + L_mask*L

            x = x.T
            x = torch.matmul(U,x)
            x = torch.matmul(L,x)
            #x = torch.gather(x, self.invP)
            x = torch.transpose(x,0,1)

            log_abs_det_jacobian = torch.log(torch.abs(U_diag))

            if logdet is not None:
                return x, logdet + log_abs_det_jacobian.expand_as(x).sum(dim=-1, keepdims=True)
            return x

        else:
            # invP*L*U*x
            LU = self.LU_init + self.LU

            # upper-triangular matrix
            U = torch.triu(LU) 

            # diagonal line
            U_diag = torch.diagonal(U) 

            # trainable mask for U
            U_mask = torch.triu(self.ones_mat)
            # U_mask = (torch.triu(self.ones_mat) >= 1)
            U = ((1-U_mask)*U).detach() + U_mask*U 

            # lower-triangular matrix
            # I = torch.eye(self.input_size)
            L = torch.tril(self.I+LU)-torch.diagonal(LU).diag()

            # trainable mask for L
            L_mask = torch.tril(self.ones_mat) - self.I
            # L_mask = (torch.tril(self.ones_mat) - torch.diagonal(self.ones_mat) >= 1)
            L = ((1-L_mask)*L).detach() + L_mask*L#entry_stop_gradients(L, L_mask)

            x = torch.transpose(x,0,1)
            #x = torch.gather(x, self.P)
            x = torch.matmul(torch.inverse(L), x)
            x = torch.matmul(torch.inverse(U), x)

            #x = torch.linalg.triangular_solve(L, x, lower=True)
            #x = torch.linalg.triangular_solve(U, x, lower=False)
            x = torch.transpose(x,0,1)
            log_abs_det_jacobian = -torch.log(torch.abs(U_diag))

            if logdet is not None:
                return x, logdet + log_abs_det_jacobian.expand_as(x).sum(dim=-1, keepdims=True)
            return x


class squeezing(torch.nn.Module):
    def __init__(self, n_dim, n_cut=1):
        super(squeezing, self).__init__()
        self.n_dim = n_dim
        self.n_cut = n_cut
        self.x = None

    def forward(self, inputs, reverse=False):
        z = inputs
        n_length = z.shape[-1]

        if not reverse:
            if n_length < self.n_cut:
                raise Exception()

            if self.n_dim == n_length:
                if self.n_dim > 2 * self.n_cut:
                    if self.x is not None:
                        raise Exception()
                    else:
                        self.x = z[:, (n_length - self.n_cut):]
                        z = z[:, :(n_length - self.n_cut)]
                else:
                    self.x = None
            elif (n_length - self.n_cut) <= self.n_cut:
                z = torch.cat((z, self.x), 1)
                self.x = None
            else:
                cut = z[:, (n_length - self.n_cut):]
                self.x = torch.cat((cut, self.x), 1)
                z = z[:, :(n_length - self.n_cut)]
        else:
            if self.n_dim == n_length:
                n_start = self.n_dim % self.n_cut
                if n_start == 0:
                    n_start += self.n_cut
                self.x = z[:, n_start:]
                z = z[:, :n_start]

            x_length = self.x.shape[-1]
            if x_length < self.n_cut:
                raise Exception()

            cut = self.x[:, :self.n_cut]
            z = torch.cat((z, cut), 1)

            if (x_length - self.n_cut) == 0:
                self.x = None
            else:
                self.x = self.x[:, self.n_cut:]

        return z
    

class scale_and_CDF(torch.nn.Module):
  def __init__(self, n_dim, n_bins=16):
    super(scale_and_CDF, self).__init__()
    self.n_dim = n_dim 
    self.n_bins = n_bins

    self.scale_layer = actnorm(n_dim)
    self.cdf_layer = CDF_quadratic(self.n_dim, self.n_bins)

  def forward(self, inputs, logdet=None, reverse=False):
    z = inputs
    assert z.shape[-1] == self.n_dim 
    if not reverse:
      z = self.scale_layer(z, logdet)
      if logdet is not None:
        z, logdet = z

      z = self.cdf_layer(z, logdet)
      if logdet is not None:
        z, logdet = z
    else:
      z = self.cdf_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

      z = self.scale_layer(z, logdet, reverse=True)
      if logdet is not None:
        z, logdet = z

    if logdet is not None:
      return z, logdet

    return z

  def actnorm_data_initialization(self):
    self.scale_layer.reset_data_initialization()



# mapping defined by a piecewise quadratic cumulative distribution function (CDF)
# Assume that each dimension has a compact support [0,1]
# CDF(x) maps [0,1] to [0,1], where the prior uniform distribution is defined.
# Since x is defined on (-inf,+inf), we only consider a CDF() mapping from
# the interval [-bound, bound] to [-bound, bound], and leave alone other points.
# The reason we do not consider a mapping from (-inf,inf) to (0,1) is the
# singularity induced by the mapping.
class CDF_quadratic(torch.nn.Module):
    def __init__(self, n_dim, n_bins, r=1.2, bound=30.0, beta=1e-8):
        super(CDF_quadratic, self).__init__()

        assert n_bins % 2 == 0

        self.n_dim = n_dim 
        self.n_bins = n_bins

        # generate a nonuniform mesh symmetric to zero,
        # and increasing by ratio r away from zero.
        self.bound = bound
        self.r = r
        self.beta = beta

        m = n_bins/2
        x1L = bound*(r-1.0)/(r**m-1.0)

        index = torch.reshape(torch.arange(0, self.n_bins+1, dtype=torch.float32),(-1,1))
        index -= m
        xr = torch.where(index>=0, (1.-torch.pow(r, index))/(1.-r), (1.-torch.pow(r,torch.abs(index)))/(1.-r))
        xr = torch.where(index>=0, x1L*xr, -x1L*xr)
        xr = torch.reshape(xr,(-1,1))
        xr = (xr + bound)/2.0/bound

        self.x1L = x1L/2.0/bound
        mesh = torch.cat([torch.reshape(torch.tensor([0.0]),(-1,1)), torch.reshape(xr[1:-1,0],(-1,1)), torch.reshape(torch.tensor([1.0]),(-1,1))],0) 
        self.register_buffer('mesh', mesh)
        elmt_size = torch.reshape(self.mesh[1:] - self.mesh[:-1],(-1,1))
        self.register_buffer('elmt_size', elmt_size)

        self.n_length = n_dim 
        self.p = torch.nn.Parameter(torch.zeros(self.n_bins-1, self.n_length))

    def forward(self, inputs, logdet=None, reverse=False):

        assert inputs.shape[-1] == self.n_dim 

        # normalize the PDF
        self._pdf_normalize()

        x = inputs
        if not reverse:
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound

            # for the interval (a,inf)
            x = torch.where(x > self.bound, self.beta * (x - self.bound) + self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet > self.bound, self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), [1], keepdims=True)
                logdet += dlogdet

            # for the interval (-inf,a)
            x = torch.where(x < -self.bound, self.beta * (x + self.bound) - self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet < -self.bound, self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), [1], keepdims=True)
                logdet += dlogdet
        else:
            # rescale such points in [-bound, bound] will be mapped to [0,1]
            x = (x + self.bound) / 2.0 / self.bound

            # cdf mapping
            x = self._cdf_inv(x, logdet)
            if logdet is not None:
                x, logdet = x

            # maps [0,1] back to [-bound, bound]
            x = x * 2.0 * self.bound - self.bound

            # for the interval (a,inf)
            x = torch.where(x > self.bound, (x - self.bound) / self.beta + self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet > self.bound, 1.0 / self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), [1], keepdims=True)
                logdet += dlogdet

            # for the interval (-inf,a)
            x = torch.where(x < -self.bound, (x + self.bound) / self.beta - self.bound, x)
            if logdet is not None:
                dlognet = x
                dlogdet = torch.where(dlognet < -self.bound, 1.0 / self.beta, 1.0)
                dlogdet = torch.sum(torch.log(dlogdet), [1], keepdims=True)
                logdet += dlogdet

        if logdet is not None:
            return x, logdet

        return x

    # normalize the piecewise representation of pdf
    def _pdf_normalize(self):
        # peicewise pdf
        p0 = torch.ones(1, self.n_length, device=self.p.device) * self.beta
        self.pdf = p0
        # self.mesh = self.mesh.to(self.p.device)
        # self.elmt_size = self.elmt_size.to(self.p.device) 

        px = torch.exp(self.p)*(self.elmt_size[:-1]+self.elmt_size[1:])/2.0
        px = (1.0 - (self.elmt_size[0] + self.elmt_size[-1]) * self.beta / 2.0) /torch.sum(px, 0, keepdim=True)
        px = px*torch.exp(self.p)
        self.pdf = torch.cat([self.pdf, px], 0)
        self.pdf = torch.cat([self.pdf, p0], 0)

        # probability in each element
        cell = (self.pdf[:-1,:] + self.pdf[1:,:])/2.0*self.elmt_size
        # CDF - contribution from previous elements.
        r_zeros= torch.zeros(1, self.n_length, device=self.p.device) 
        self.F_ref = r_zeros
        for i in range(1, self.n_bins):
            tp = torch.sum(cell[:i,:], 0, keepdim=True)
            self.F_ref = torch.cat([self.F_ref, tp], 0)

    # the cdf is a piecewise quadratic function.
    def _cdf(self, x, logdet=None):

        xr = torch.broadcast_to(self.mesh, [self.n_bins + 1, self.n_length])
        k_ind = torch.searchsorted(xr.T.contiguous(), x.T.contiguous(), right=True)
        k_ind = k_ind.T.to(torch.int64)
        k_ind -= 1


        cover = torch.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        v1 = torch.reshape(self.pdf[:,0][k_ind[:,0]],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.pdf[:,i][k_ind[:,i]],(-1,1))
            v1 = torch.cat([v1, tp], 1)

        v2 = torch.reshape(self.pdf[:,0][k_ind[:,0]+1],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.pdf[:,i][k_ind[:,i]+1],(-1,1))
            v2 = torch.cat([v2, tp], 1)

        xmodi = torch.reshape(x[:,0] - self.mesh[:,0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(x[:,i] - self.mesh[:,0][k_ind[:, i]], (-1, 1))
            xmodi = torch.cat([xmodi, tp], 1)

        h_list = torch.reshape(self.elmt_size[:,0][k_ind[:,0]],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.elmt_size[:,0][k_ind[:,i]],(-1,1))
            h_list = torch.cat([h_list, tp], 1)

        F_pre = torch.reshape(self.F_ref[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.F_ref[:, i][k_ind[:, i]], (-1, 1))
            F_pre = torch.cat([F_pre, tp], 1)

        y = torch.where(cover>0, F_pre + xmodi**2/2.0*(v2-v1)/h_list + xmodi*v1, x)
       
        if logdet is not None:
            dlogdet = torch.where(cover > 0, xmodi * (v2 - v1) / h_list + v1, 1.0)
            dlogdet = torch.sum(torch.log(dlogdet), axis=[1], keepdim=True)
            return y, logdet + dlogdet

        return y

    # inverse of the cdf
    def _cdf_inv(self, y, logdet=None):
        xr = torch.broadcast_to(self.mesh, [self.n_bins+1, self.n_length])
        yr1 = self._cdf(xr)

        p0 = torch.zeros(1, self.n_length, device=self.p.device)
        p1 = torch.ones(1, self.n_length, device=self.p.device)
        yr = torch.cat([p0, yr1[1:-1,:], p1], 0)

        k_ind = torch.searchsorted((yr.T).contiguous(), (y.T).contiguous(), right=True)
        k_ind = k_ind.T 
        k_ind = k_ind.to(torch.int64)
        k_ind -= 1

        cover = torch.where(k_ind*(k_ind-self.n_bins+1)<=0, 1.0, 0.0)

        k_ind = torch.where(k_ind < 0, 0, k_ind)
        k_ind = torch.where(k_ind > (self.n_bins-1), self.n_bins-1, k_ind)

        c_cover = torch.reshape(cover[:,0], (-1,1))
        v1 = torch.where(c_cover > 0, torch.reshape(self.pdf[:,0][k_ind[:,0]],(-1,1)), -1.0)
        for i in range(1, self.n_length):
            c_cover = torch.reshape(cover[:, i], (-1,1))
            tp = torch.where(c_cover > 0, torch.reshape(self.pdf[:,i][k_ind[:,i]],(-1,1)), -1.0)
            v1 = torch.cat([v1, tp], 1)

        c_cover = torch.reshape(cover[:,0], (-1,1))
        v2 = torch.where(c_cover > 0, torch.reshape(self.pdf[:,0][k_ind[:,0]+1],(-1,1)), -2.0)
        for i in range(1, self.n_length):
            c_cover = torch.reshape(cover[:,i], (-1,1))
            tp = torch.where(c_cover > 0, torch.reshape(self.pdf[:,i][k_ind[:,i]+1],(-1,1)), -2.0)
            v2 = torch.cat([v2, tp], 1)

        ys = torch.reshape(y[:, 0] - yr[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(y[:, i] - yr[:, i][k_ind[:, i]], (-1, 1))
            ys = torch.cat([ys, tp], 1)

        xs = torch.reshape(xr[:, 0][k_ind[:, 0]], (-1, 1))
        for i in range(1, self.n_length):
            tp = torch.reshape(xr[:, i][k_ind[:, i]], (-1, 1))
            xs = torch.cat([xs, tp], 1)

        h_list = torch.reshape(self.elmt_size[:,0][k_ind[:,0]],(-1,1))
        for i in range(1, self.n_length):
            tp = torch.reshape(self.elmt_size[:,0][k_ind[:,i]],(-1,1))
            h_list = torch.cat([h_list, tp], 1)

        h_s = (v2 - v1) / h_list
        tp = v1 * v1 + 2.0 * ys * h_s
        tp = torch.sqrt(tp) + v1
        tp = 2.0 * ys / tp
        tp += xs

        x = torch.where(cover > 0, tp, y)

        if logdet is not None:
            tp = 2.0 * ys * h_s
            tp += v1 * v1
            tp = 1.0 / torch.sqrt(tp)

            dlogdet = torch.where(cover > 0, tp, 1.0)
            dlogdet = torch.sum(torch.log(dlogdet), [1], keepdims=True)
            return x, logdet + dlogdet

        return x