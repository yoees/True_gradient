import torch
import numpy as np 
import matplotlib.pyplot as plt
from grad_eval_functions import *
from functions import *

th, expo, mant = 0.3, 8, 23
#umax_exp, umin_exp = exp_lut(-15, 8)
umax_exp, umin_exp, lut_exp = exp_lut(-15, 8)
Iset, umin, umax = grad_lut(th, expo, mant)

u = 4*torch.rand(100,100,32,32).cuda()-2
idx = grad_idx(u, umax, umin, th, 4).long() 
I = Iset[idx]
#s = s_eval(u, th, expo, mant, umax_exp, umin_exp, 4) 
s = s_eval(u, th, expo, mant, umax_exp, umin_exp, 4, lut_exp) 
grad_u = (I*s).sum(-1)

plt.scatter(u.reshape(-1).cpu(),grad_u.reshape(-1).cpu())
plt.show()
