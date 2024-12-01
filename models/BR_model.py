from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

import models.BR_layers as BR_layers
import models.BR_data as BR_data 



# invertible mapping based on real NVP and KR rearrangement and CDF inverse
class IM_rNVP_KR_CDF(torch.nn.Module):
    def __init__(self, 
                 n_dim, 
                 n_step, 
                 n_depth,
                 n_width=32,
                 shrink_rate=1.,
                 n_bins=32,
                 rotation=True):
        super(IM_rNVP_KR_CDF, self).__init__()

        # two affine coupling layers are needed for each update of the vector
        assert n_depth % 2 == 0

        self.n_dim = n_dim # dimension of the data
        self.n_step = n_step # step size for dimension reduction
        self.n_depth = n_depth # depth for flow_mapping
        self.n_width = n_width
        self.n_bins = n_bins
        self.shrink_rate = shrink_rate
        self.rotation = rotation

        # the number of filtering stages
        self.n_stage = n_dim // n_step 
        if n_dim % n_step == 0:
            self.n_stage -= 1

        n_length = n_dim 

        # the number of rotation layers
        self.n_rotation = self.n_stage

        if rotation:
            self.rotations = torch.nn.ModuleList([BR_layers.W_LU(n_dim)])
            for i in range(1, self.n_rotation):
                # rotate the coordinate system for a better representation of data
                self.rotations.append(BR_layers.W_LU(n_dim-i*n_step))

        # flow mapping with n_stage
        self.flow_mappings = torch.nn.ModuleList() 
        for i in range(self.n_stage):
            # flow_mapping given by such as real NVP
            n_split_at = n_dim - (i+1) * n_step
            self.flow_mappings.append(BR_layers.flow_mapping(n_length, 
                                                             n_depth,
                                                             n_split_at,
                                                             n_width=n_width))
            n_width = int(n_width*self.shrink_rate)
            n_length = n_length - n_step 

        # data will pass the squeezing layer at the end of each stage
        self.squeezing_layer = BR_layers.squeezing(n_dim, n_step)

        if self.n_bins > 0:
            self.cdf_layer = BR_layers.scale_and_CDF(n_dim, n_bins)

        # the prior distribution is the standard Gaussian
        self.log_prior = BR_data.log_standard_Gaussian

    # computing the logarithm of the estimated pdf on the input data.
    def forward(self, inputs):
        objective = torch.zeros_like(inputs)[:,0]
        objective = torch.reshape(objective, [-1,1])

        # f(y) and log of jacobian
        z, objective = self.mapping_to_prior(inputs, objective)

        # logrithm of estimated pdf
        objective += self.log_prior(z)

        return objective

    # mapping from data to prior
    def mapping_to_prior(self, inputs, logdet=None):
        z = inputs

        for i in range(self.n_stage):
            if logdet is not None:
                if self.rotation and i < self.n_rotation:
                    z, logdet = self.rotations[i](z, logdet)
                
                z, logdet = self.flow_mappings[i](z, logdet)
            else:
                if self.rotation and i < self.n_rotation:
                    z = self.rotations[i](z)

                z = self.flow_mappings[i](z)
            z = self.squeezing_layer(z)

        if self.n_bins > 0:
            z = self.cdf_layer(z, logdet)
            if logdet is not None:
                z, logdet = z

        if logdet is not None:
            return z, logdet
        else:
            return z

    # mapping from prior to data
    def mapping_from_prior(self, inputs):
        z = inputs

        if self.n_bins > 0:
            z = self.cdf_layer(z, reverse=True)
        
        for i in reversed(range(self.n_stage)):
            z = self.squeezing_layer(z, reverse=True)
            z = self.flow_mappings[i](z, reverse=True)

            if self.rotation and i < self.n_rotation:
                z = self.rotations[i](z, reverse=True)

        # generate samples in domain [lb, hb]^d
        # z = self.bounded_support_layer(z, reverse=True)
        return z

    # data initialization for actnorm layers
    def actnorm_data_initialization(self):
        for i in range(self.n_stage):
            self.flow_mappings[i].actnorm_data_initialization()
