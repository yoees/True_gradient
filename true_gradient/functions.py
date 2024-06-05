import torch
import numpy as np
from grad_eval_functions import *
import torch.nn as nn 

dtype = torch.float32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.set_printoptions(precision=15)

def lif(ops, x, u, tau_m, thresh, type_grad, lens, spike, dt, Iset, umax, umin, umax_exp, umin_exp, expo, mant, lut_exp, grad_max):    
    batch_size = x.shape[0]
    u = u * np.exp(-dt / tau_m) * (1 - spike) + ops(x)
    if type_grad == 'STBP':
        spike = dsdu.apply(u, thresh, lens)
    elif type_grad == 'true_grad':
        spike = dsdu_true.apply(u, thresh, Iset, umax, umin, umax_exp, umin_exp, expo, mant, lut_exp, grad_max)
    return u, spike

def srm(ops, x, um, us, tau_m, tau_s, thresh, type_grad, lens, spike, dt, Iset, umax, umin, umax_exp, umin_exp, expo, mant, lut_exp, grad_max, grad_approx):
    const = tau_m / (tau_m - tau_s)
    um = um * np.exp(-dt / tau_m) * (1 - spike) + ops(x)  
    us = us * np.exp(-dt / tau_s) * (1 - spike) + ops(x)
    u = (um - us)*const    
    if type_grad == 'STBP':
        spike = dsdu.apply(u, thresh, lens)
    elif type_grad == 'true_grad':
        spike = dsdu_true.apply(u, thresh, Iset, umax, umin, umax_exp, umin_exp, expo, mant, lut_exp, grad_max)
    elif type_grad == 'true_grad_approx':
        spike = dsdu_true_approx.apply(u, thresh, Iset, umax, umin, grad_max, grad_approx)
    return um, us, spike

class dsdu(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, u, thresh, lens):
        ctx.save_for_backward(u)
        ctx.thresh = thresh
        ctx.lens = lens
        return u.gt(thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        grad_input = grad_output.clone()
        thresh = ctx.thresh
        lens = ctx.lens
        temp = abs(u - thresh) < lens
        grad_u = grad_input * temp.float() 
        # print(grad_u.max())
        return grad_u, None, None

class dsdu_true(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, u, thresh, Iset, umax, umin, umax_exp, umin_exp, expo, mant, lut_exp, grad_max):
        ctx.save_for_backward(u)
        ctx.thresh = thresh
        ctx.Iset = Iset
        ctx.umax = umax
        ctx.umin = umin
        ctx.expo = expo
        ctx.mant = mant
        ctx.umax_exp = umax_exp
        ctx.umin_exp = umin_exp
        ctx.lut_exp = lut_exp
        ctx.grad_max = grad_max
        return u.gt(thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        grad_input = grad_output.clone()
        thresh = ctx.thresh
        Iset = ctx.Iset 
        umax = ctx.umax
        umin = ctx.umin
        expo = ctx.expo
        mant = ctx.mant
        umax_exp = ctx.umax_exp
        umin_exp = ctx.umin_exp
        lut_exp = ctx.lut_exp
        grad_max = ctx.grad_max
        dim = u.dim()               
        idx = grad_idx(u, umax, umin, thresh, dim).long()       
        I = Iset[idx]       
        s = s_eval(u, thresh, expo, mant, umax_exp, umin_exp, dim, lut_exp)

        grad_temp = torch.clamp((I*s).sum(-1), 0, grad_max)/grad_max
        # if dim == 2:
        #     d0, d1 = grad_temp.size()
        #     grad_temp = nn.functional.normalize(grad_temp.reshape(d0,-1),p=1,dim=1)
        #     grad_temp = grad_temp.reshape(d0, d1)            
        # else:
        #     d0, d1, d2, d3 = grad_temp.size()
        #     grad_temp = nn.functional.normalize(grad_temp.reshape(d0,-1),p=1,dim=1)
        #     grad_temp = grad_temp.reshape(d0, d1, d2, d3)

        grad_u = grad_input * grad_temp
        # grad_u = (grad_input * torch.clamp((I*s).sum(-1), 0, 100))
        return grad_u, None,None,None,None,None,None,None,None,None,None

class dsdu_true_approx(torch.autograd.Function):    
    @staticmethod
    def forward(ctx, u, thresh, Iset, umax, umin, grad_max, grad_approx):
        ctx.save_for_backward(u)
        ctx.thresh = thresh
        ctx.Iset = Iset
        ctx.umax = umax
        ctx.umin = umin
        ctx.grad_max = grad_max
        ctx.grad_approx = grad_approx
        return u.gt(thresh).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        grad_input = grad_output.clone()
        thresh = ctx.thresh
        Iset = ctx.Iset 
        umax = ctx.umax
        umin = ctx.umin
        grad_max = ctx.grad_max
        grad_approx = ctx.grad_approx
        dim = u.dim()               
        idx = grad_idx(u, umax, umin, thresh, dim).long()       
        grad_temp = torch.clamp(grad_approx[idx], 0, grad_max)/grad_max
        # print(grad_approx[idx].min())
        # if dim == 2:
        #     d0, d1 = grad_temp.size()
        #     grad_temp = nn.functional.normalize(grad_temp.reshape(d0,-1),p=1,dim=1)
        #     grad_temp = grad_temp.reshape(d0, d1)            
        # else:
        #     d0, d1, d2, d3 = grad_temp.size()
        #     grad_temp = nn.functional.normalize(grad_temp.reshape(d0,-1),p=1,dim=1)
        #     grad_temp = grad_temp.reshape(d0, d1, d2, d3)

        grad_u = grad_input * grad_temp
        # grad_u = (grad_input * torch.clamp((I*s).sum(-1), 0, 100))
        return grad_u, None,None,None,None,None,None

def get_layer_shape(input_shape, kernel_size, stride, padding):
    feature_height = int((input_shape[0] + 2*padding - kernel_size)//stride + 1) 
    feature_width = int((input_shape[1] + 2*padding - kernel_size)//stride + 1)
    return [feature_height, feature_width]

def grad_idx(u, umax, umin, thresh, dim):
    if dim == 4:        
        u = u[:,:,:,:,None].repeat(1,1,1,1,umax.size()[0]).type(dtype).to(device)
        idx_temp = torch.where((u < umax) & (u >= umin) & (u<thresh), torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))
        idx_temp = torch.where((u <= umax) & (u > umin) & (u>=thresh), torch.Tensor([1]).to(device), idx_temp)
        if idx_temp.sum() != u[:,:,:,:,0].numel():
            print('GRAD_IDX matching error in dim 4!!!', u.min(), u.max())
        idx = torch.argmax(idx_temp, dim=4)
    elif dim == 2: 
        u = u[:,:,None].repeat(1,1,umax.size()[0]).type(dtype).to(device)
        idx_temp = torch.where((u < umax) & (u >= umin) & (u<thresh), torch.Tensor([1]).to(device), torch.Tensor([0]).to(device))
        idx_temp = torch.where((u <= umax) & (u > umin) & (u>=thresh), torch.Tensor([1]).to(device), idx_temp)
        if idx_temp.sum() != u[:,:,0].numel():
            print('GRAD_IDX matching error in dim 2!!!')            
        idx = torch.argmax(idx_temp, dim=2)
    else:
        print('Check dimmension!!!')
    return idx

def s_eval(u, thresh, expo, mant, umax, umin, dim, lut_exp):
    l = mant + expo + 1  
    if dim == 4:
        u_exp = u[:,:,:,:,None].repeat(1,1,1,1,umax.size()[0]).type(dtype).to(device)
        idx_temp = ((abs(u_exp) < umax) & (abs(u_exp) >= umin)).float()
        if idx_temp.sum() != u.numel():
            print('IDX matching error in 4D s_eval!!!')
        idx = torch.argmax(idx_temp, dim=4).long()
        exp_array = lut_exp[idx].type(dtype)
        
        s0 = digit2_cuda(0, l-1, mant)[:,None,None,None,None].repeat(1,u.size()[0], u.size()[1], u.size()[2], u.size()[3]).type(dtype).to(device)
        u_exp = u[None,:,:,:,:].repeat(l,1,1,1,1).type(dtype).to(device)
        s0 = torch.where((s0 < 1) & (u_exp < 0), -s0*exp_array, s0)
        s0 = torch.where((s0 < 1) & (u_exp >= 0), s0*exp_array, s0)
        s0 = torch.where((s0 >= 1) & (s0 < 2**(l-2-mant)) & (u_exp < thresh), u_exp*(2**(s0)-1), s0)
        s0 = torch.where((s0 >= 1) & (s0 < 2**(l-2-mant)) & (u_exp >= thresh), u_exp*(1-2**(-s0)), s0) 
        s0 = torch.where(s0 == 2**(l-2-mant), torch.Tensor([0]).to(device), s0) # ok
        s0 = torch.where((s0  == 2**(l-1-mant)) & (u_exp < 0), 2*u_exp, s0) # ok
        s0 = torch.where((s0  == 2**(l-1-mant)) & (u_exp >= 0), -2*u_exp, s0) # ok
        s = torch.where(s0 == 0, torch.Tensor([0]).to(device), 1/s0)   
        return s.permute(1,2,3,4,0)             

    elif dim == 2:        
        u_exp = u[:,:,None].repeat(1,1,umax.size()[0]).type(dtype).to(device)
        idx_temp = ((abs(u_exp) < umax) & (abs(u_exp) >= umin)).float()
        if idx_temp.sum() != u.numel():
            print('IDX matching error in 2D s_eval!!!', u.numel() - idx_temp.sum())            
        idx = torch.argmax(idx_temp, dim=2).long()
        exp_array = lut_exp[idx].type(dtype)
        
        s0 = digit2_cuda(0, l-1, mant)[:,None,None].repeat(1, u.size()[0], u.size()[1]).type(dtype).to(device)
        u_exp = u[None,:,:].repeat(l,1,1).to(device)    
        s0 = torch.where((s0 < 1) & (u_exp < 0), -s0*exp_array, s0)
        s0 = torch.where((s0 < 1) & (u_exp >= 0), s0*exp_array, s0)
        s0 = torch.where((s0 >= 1) & (s0 < 2**(l-2-mant)) & (u_exp < thresh), u_exp*(2**(s0)-1), s0)
        s0 = torch.where((s0 >= 1) & (s0 < 2**(l-2-mant)) & (u_exp >= thresh), u_exp*(1-2**(-s0)), s0)
        s0 = torch.where(s0 == 2**(l-2-mant), torch.Tensor([0]).to(device), s0)
        s0 = torch.where((s0  == 2**(l-1-mant)) & (u_exp < 0), 2*u_exp, s0)
        s0 = torch.where((s0  == 2**(l-1-mant)) & (u_exp >= 0), -2*u_exp, s0)
        s = torch.where(s0 == 0, torch.Tensor([0]).to(device), 1/s0)
        return s.permute(1,2,0)

    elif dim == 1:
        u_exp = u[:,None].repeat(1,umax.size()[0]).type(dtype).to(device)
        idx_temp = ((abs(u_exp) < umax) & (abs(u_exp) >= umin)).float()
        if idx_temp.sum() != u.numel():
            print('IDX matching error in 1D s_eval!!!')            
        idx = torch.argmax(idx_temp, dim=1).long()
        exp_array = lut_exp[idx].type(dtype)
        
        s0 = digit2_cuda(0, l-1, mant)[:,None].repeat(1, u.size()[0]).type(dtype).to(device)
        u_exp = u[None,:].repeat(l,1).to(device)
        s0 = torch.where(s0 < 1, s0*exp_array, s0)        
        s0 = torch.where((s0 < 1) & (u_exp < 0), -s0, s0)        
        s0 = torch.where((s0 >= 1) & (s0 < 2**(l-2-mant)) & (u_exp < thresh), u_exp*(2**(s0)-1), s0)
        s0 = torch.where((s0 >= 1) & (s0 < 2**(l-2-mant)) & (u_exp >= thresh), u_exp*(1-2**(-s0)), s0)
        s0 = torch.where(s0 == 2**(l-2-mant), torch.Tensor([0]).to(device), s0)
        s0 = torch.where((s0  == 2**(l-1-mant)) & (u_exp < 0), 2*u_exp, s0)
        s0 = torch.where((s0  == 2**(l-1-mant)) & (u_exp >= 0), -2*u_exp, s0)
        s = torch.where(s0 == 0, torch.Tensor([0]).to(device), 1/s0)
        return s.permute(1,0)
    else:
        print('Check dimmension!!!')
    


