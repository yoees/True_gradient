import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functions import *
from grad_eval_functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float

class srm_snn(nn.Module):
    def __init__(self, dataset, thresh, tau_m, tau_s, frate, dt, input_shape, 
                num_conv_layers, kernel_size, channels, padding, pool_size, stride, 
                num_dense_layers, n_dense, type_grad, lens, expo, mant, expo_lim1, expo_lim2, grad_max):
        super(srm_snn, self).__init__()
        
        self.dataset = dataset
        self.thresh = thresh
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.frate = frate
        self.dt = dt
        self.input_shape = input_shape
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        self.pool_size = pool_size
        self.stride = stride
        self.num_dense_layers = num_dense_layers
        self.n_dense = n_dense
        self.type_grad = type_grad       
        self.lens = lens
        self.expo = expo
        self.mant = mant
        self.expo_lim1 = expo_lim1
        self.expo_lim2 = expo_lim2
        self.grad_max = grad_max

        self.sn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.num_layers = self.num_conv_layers + self.num_dense_layers

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]
        if self.num_conv_layers == 0:
            feature_length = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        else:
            feature_length = self.build_conv_stack(self.channels, feature_height, feature_width, 
                                               self.padding, self.pool_size, self.kernel_size, self.stride)
        self.build_dense_stack([feature_length], self.n_dense)        

        # Weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight, 0)

        self.sn_layers.apply(init_weights)

        # True_grad LUT initialization
        self.Iset, self.umin, self.umax = grad_lut(self.thresh, self.expo, self.mant)
        self.umax_exp, self.umin_exp, self.lut_exp = exp_lut(self.expo_lim1, self.expo_lim2)
        # For True_grad_approx        
        self.grad_approx = grad_u_approx(self.Iset, self.umax, self.thresh, self.expo, self.mant)        

    def forward(self, input, batch_size, num_steps, limit):        
        vm = []
        vs = []        
        spike = []
        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]
        for i in range(self.num_conv_layers):  
            feature_height, feature_width = get_layer_shape([feature_height, feature_width], 
                                                            kernel_size=self.kernel_size[i],
                                                            stride=self.stride[i],
                                                            padding=self.padding[i])          
            vm.append(torch.zeros(batch_size, self.channels[i+1], feature_height, feature_width, dtype=dtype, device=device))
            vs.append(torch.zeros(batch_size, self.channels[i+1], feature_height, feature_width, dtype=dtype, device=device))
            spike.append(torch.zeros(batch_size, self.channels[i+1], feature_height, feature_width, dtype=dtype, device=device))
            
            if i >= 1: # for new network
                feature_height //= self.pool_size[i-1]
                feature_width //= self.pool_size[i-1]

        if self.num_conv_layers == 0:
            feature_length = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        else:
            feature_length = self.channels[-1]*feature_height*feature_width
        
        for i in range(self.num_dense_layers):
            feature_length = self.n_dense[i]
            vm.append(torch.zeros(batch_size, feature_length, dtype=dtype, device=device))
            vs.append(torch.zeros(batch_size, feature_length, dtype=dtype, device=device))
            spike.append(torch.zeros(batch_size, feature_length, dtype=dtype, device=device))           
        
        out_spike_sum = torch.zeros(batch_size, feature_length, dtype=dtype, device=device)       
        for step in range(num_steps):
            # MNIST input encoding : Poisson spike generation            
            if self.dataset == 'F-MNIST':
                in_spike = (input * self.frate > torch.rand(input.size(), dtype=dtype, device=device)).float()                          
            # N-MNIST input encoding
            elif self.dataset == 'DVSGesture':
                if limit == 'yes':
                    in_spike = torch.clamp(input[:, step, :,:,:].float(), 0, 1)
                else:
                    in_spike = input[:, step, :,:,:].float()

            if self.num_conv_layers == 0:
                in_spike = in_spike.view(batch_size,-1)           

            for i in range(self.num_layers):
                # base_layer = self.noso_layers[i]                
                vm[i], vs[i], spike[i] = srm(self.sn_layers[i],
                                            in_spike,
                                            vm[i],
                                            vs[i], 
                                            self.tau_m,
                                            self.tau_s, 
                                            self.thresh,
                                            self.type_grad,
                                            self.lens,
                                            spike[i],
                                            self.dt,
                                            self.Iset,
                                            self.umax,
                                            self.umin,
                                            self.umax_exp,
                                            self.umin_exp,
                                            self.expo,
                                            self.mant,
                                            self.lut_exp,
                                            self.grad_max,
                                            self.grad_approx)                
                if 0 < i < self.num_conv_layers-1: # for new network
                    pool = self.pool_layers[i-1] # for new network
                    in_spike = pool(spike[i])
                elif i == self.num_conv_layers-1:
                    pool = self.pool_layers[i-1] # for new network
                    in_spike = pool(spike[i]).view(batch_size,-1)
                else:
                    in_spike = spike[i]
            out_spike_sum += spike[-1]                                    
            
        return  out_spike_sum/num_steps

    def build_conv_stack(self, channels, feature_height, feature_width, padding, pool_size, kernel_size, stride):        
        for i in range(self.num_conv_layers):
            if i == 0:  ## for new network
                layer = nn.Conv2d(channels[i], channels[i+1], kernel_size[i], stride[i], padding[i], bias=False)

                self.sn_layers.append(layer)
            else:
                layer = nn.Conv2d(channels[i], channels[i+1], kernel_size[i], stride[i], padding[i], bias=False)
                pool = nn.MaxPool2d(kernel_size=pool_size[i-1]) # for new network

                self.sn_layers.append(layer)
                self.pool_layers.append(pool)

            feature_height, feature_width = get_layer_shape([feature_height, feature_width], 
                                                            kernel_size=kernel_size[i],
                                                            stride=stride[i],
                                                            padding=padding[i])
            if i >= 1: # for new network                                              
                feature_height //= pool_size[i-1]
                feature_width //= pool_size[i-1]
        return channels[-1]*feature_height*feature_width

    def build_dense_stack(self, feature_input, n_dense):
        feature_length = feature_input + n_dense
        for i in range(self.num_dense_layers):
            layer = nn.Linear(feature_length[i], feature_length[i+1], bias=False)
            self.sn_layers.append(layer)

    
class lif_snn(nn.Module):
    def __init__(self, dataset, thresh, tau_m, tau_s, frate, dt, input_shape, 
                num_conv_layers, kernel_size, channels, padding, pool_size, stride, 
                num_dense_layers, n_dense, type_grad, lens, expo, mant, expo_lim1, expo_lim2, grad_max):
        super(lif_snn, self).__init__()
        
        self.dataset = dataset
        self.thresh = thresh
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.frate = frate
        self.dt = dt
        self.input_shape = input_shape
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.channels = channels
        self.padding = padding
        self.pool_size = pool_size
        self.stride = stride
        self.num_dense_layers = num_dense_layers
        self.n_dense = n_dense
        self.type_grad = type_grad       
        self.lens = lens
        self.expo = expo
        self.mant = mant
        self.expo_lim1 = expo_lim1
        self.expo_lim2 = expo_lim2
        self.grad_max = grad_max

        self.sn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.num_layers = self.num_conv_layers + self.num_dense_layers

        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]
        if self.num_conv_layers == 0:
            feature_length = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        else:
            feature_length = self.build_conv_stack(self.channels, feature_height, feature_width, 
                                               self.padding, self.pool_size, self.kernel_size, self.stride)
        self.build_dense_stack([feature_length], self.n_dense)        

        # Weight initialization
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                # nn.init.constant_(m.weight, 0)

        self.sn_layers.apply(init_weights)

        # True_grad LUT initialization
        self.Iset, self.umin, self.umax = grad_lut(self.thresh, self.expo, self.mant)
        self.umax_exp, self.umin_exp, self.lut_exp = exp_lut(expo_lim1, expo_lim2)

    def forward(self, input, batch_size, num_steps, limit):        
        vm = []       
        spike = []
        feature_height = self.input_shape[1]
        feature_width = self.input_shape[2]
        for i in range(self.num_conv_layers):            
            # vm.append(torch.zeros(batch_size, self.channels[i], feature_height, feature_width, dtype=dtype, device=device))
            # vs.append(torch.zeros(batch_size, self.channels[i], feature_height, feature_width, dtype=dtype, device=device))
            feature_height, feature_width = get_layer_shape([feature_height, feature_width], 
                                                            kernel_size=self.kernel_size[i],
                                                            stride=self.stride[i],
                                                            padding=self.padding[i])
            # feature_height //= self.pool_size[i]
            # feature_width //= self.pool_size[i]            
            spike.append(torch.zeros(batch_size, self.channels[i+1], feature_height, feature_width, dtype=dtype, device=device))
            vm.append(torch.zeros_like(spike[-1]))
            if i >= 1: # for new network
                feature_height //= self.pool_size[i-1]
                feature_width //= self.pool_size[i-1]

        if self.num_conv_layers == 0:
            feature_length = self.input_shape[0]*self.input_shape[1]*self.input_shape[2]
        else:
            feature_length = self.channels[-1]*feature_height*feature_width
        
        for i in range(self.num_dense_layers):
            feature_length = self.n_dense[i]
            vm.append(torch.zeros(batch_size, feature_length, dtype=dtype, device=device))            
            spike.append(torch.zeros(batch_size, feature_length, dtype=dtype, device=device))
        
        out_spike_sum = torch.zeros(batch_size, feature_length, dtype=dtype, device=device)       
        for step in range(num_steps):
            # MNIST input encoding : Poisson spike generation            
            if self.dataset == 'F-MNIST':
                in_spike = (input * self.frate > torch.rand(input.size(), dtype=dtype, device=device)).float()                          
            # N-MNIST input encoding
            elif self.dataset == 'DVSGesture':
                if limit == 'yes':
                    in_spike = torch.clamp(input[:, step, :,:,:].float(), 0, 1)
                else:
                    in_spike = input[:, step, :,:,:].float()

            if self.num_conv_layers == 0:
                in_spike = in_spike.view(batch_size,-1)           

            for i in range(self.num_layers):
                # base_layer = self.noso_layers[i]   
                # print(vm[i].size(), in_spike.size())
                vm[i], spike[i] = lif(self.sn_layers[i],
                                            in_spike,
                                            vm[i],                                           
                                            self.tau_m,                                             
                                            self.thresh,
                                            self.type_grad,
                                            self.lens,
                                            spike[i],
                                            self.dt,
                                            self.Iset,
                                            self.umax,
                                            self.umin,
                                            self.umax_exp,
                                            self.umin_exp,
                                            self.expo,
                                            self.mant,
                                            self.lut_exp,
                                            self.grad_max)                
                if 0 < i < self.num_conv_layers-1: # for new network
                    pool = self.pool_layers[i-1] # for new network
                    in_spike = pool(spike[i])
                elif i == self.num_conv_layers-1:
                    pool = self.pool_layers[i-1] # for new network
                    in_spike = pool(spike[i]).view(batch_size,-1)
                else:
                    in_spike = spike[i]
            out_spike_sum += spike[-1]                                    
            
        return  out_spike_sum/num_steps

    def build_conv_stack(self, channels, feature_height, feature_width, padding, pool_size, kernel_size, stride):        
        for i in range(self.num_conv_layers):            
            if i == 0:  # for new network
                layer = nn.Conv2d(channels[i], channels[i+1], kernel_size[i], stride[i], padding[i], bias=False)

                self.sn_layers.append(layer)
            else:
                layer = nn.Conv2d(channels[i], channels[i+1], kernel_size[i], stride[i], padding[i], bias=False)
                pool = nn.MaxPool2d(kernel_size=pool_size[i-1]) # for new network

                self.sn_layers.append(layer)
                self.pool_layers.append(pool)

            feature_height, feature_width = get_layer_shape([feature_height, feature_width], 
                                                            kernel_size=kernel_size[i],
                                                            stride=stride[i],
                                                            padding=padding[i])
            if i >=1:  # for new network                                            
                feature_height //= pool_size[i-1]
                feature_width //= pool_size[i-1]
        return channels[-1]*feature_height*feature_width

    def build_dense_stack(self, feature_input, n_dense):
        feature_length = feature_input + n_dense
        for i in range(self.num_dense_layers):
            layer = nn.Linear(feature_length[i], feature_length[i+1], bias=False)
            self.sn_layers.append(layer)