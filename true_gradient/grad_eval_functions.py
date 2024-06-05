import numpy as np 
from fxpmath import Fxp
import matplotlib.pyplot as plt
import struct
import torch

dtype = torch.float32 

def digit1(a, b, c):
	x = np.arange(b,a-1,-1)
	x1 = 2**((c-x).astype(float))
	return x1

def digit2(a, b, c):
	x = np.arange(b,a-1,-1)
	x1 = 2**((x-c).astype(float))
	return x1

def digit2_cuda(a, b, c):
	x = torch.arange(b,a-1,-1).cuda()
	x1 = 2**((x-c)).type(dtype)
	return x1

def float32_bit_pattern(value):
	return sum(b << 8*i for i,b in enumerate(struct.pack('f', value)))
def int_to_binary(value, bits):
    return bin(value).replace('0b', '').rjust(bits, '0')
def conversion(x, mant, expo):
	x_bin = int_to_binary(float32_bit_pattern(x), 32)
	bwidth = 32
	# # Ruling out the sign bit	
	x_bin = np.array(list(map(int, x_bin)))
	bwidth = 32

	x_nonzero = np.nonzero(x_bin)[0]	
	x_all = bwidth - x_nonzero - 1

	x_zero = np.nonzero(x_bin==0)[0]	
	x0_all = np.sort(bwidth - x_zero - 1)

	return x_all, len(x_all), x0_all, len(x0_all), x_bin

def sig(x, expo, mant):
	x_bin = int_to_binary(float32_bit_pattern(x), 32)
	bwidth = 32
	x_bin = np.array(list(map(int, x_bin)))[expo+1:]
	return np.dot(x_bin,digit2(0, mant-1, mant))+1

def todecimal(x, expo, mant):
	x_expo = x[1:expo+1]
	x_mant = x[expo+1:]
	# print(len(x_mant))
	significand = 1 + np.dot(digit2(0, mant-1, mant),x_mant)
	exponent = 2**((np.dot(digit2(mant, mant+expo-1, mant),x_expo))-2**(expo-1)+1)
	return (significand*exponent)

def grad_lut(th, expo, mant):
	n_all, N_all, z_all, Z_all, thb = conversion(th, mant, expo)
	l = mant + expo + 1	

	I1 = np.arange(l-1, -1, -1)	
	Iset = []
	umin, umax = [], []

	# case 1: u < 0
	I = np.zeros(l)
	I[0] = 1
	Iset.append(I)
	umin.append(-1.E20)
	umax.append(-th)

	I[0] = 0
	Iset.append(I)
	umin.append(-th)
	umax.append(0)

	# case 2: 0< u < theta		
	if N_all == 1:
		I = np.where(I1>=n_all[0], 1, 0)
		I[0] = 0
		Iset.append(I)
		umin.append(0)
		umax.append(th)
		# urange.append(np.array([0, th]))
	elif N_all == 2:
		I = np.where(I1>n_all[0], 1, 0)
		I[0] = 0
		Iset.append(I)
		utemp = conversion(th, mant, expo)[4]
		utemp[l - n_all[0] - 1] = 0
		umin.append(0)
		umax.append(todecimal(utemp, expo, mant))

		I = np.where(I1>=n_all[0], 1, 0)
		I[0] = 0
		Iset.append(I)
		utemp1 = np.zeros(l)
		utemp1[l - n_all[0] - 1] = 1
		umin.append(todecimal(utemp, expo, mant))
		umax.append(todecimal(utemp1, expo, mant))

		I = np.where((I1>=n_all[1]) & (I1 != n_all[0]), 1, 0)
		I[0] = 0
		Iset.append(I)
		umin.append(todecimal(utemp1, expo, mant))
		umax.append(todecimal(thb, expo, mant))
	else:
		I = np.where(I1>n_all[0], 1, 0)
		I[0] = 0
		Iset.append(I)
		utemp = conversion(th, mant, expo)[4]		
		utemp[l - n_all[0] - 1] = 0
		umin.append(0)
		umax.append(todecimal(utemp, expo, mant))

		I = np.where(I1>=n_all[0], 1, 0)
		I[0] = 0
		Iset.append(I)
		utemp1 = np.zeros(l)
		utemp1[l - n_all[0] - 1] = 1
		umin.append(todecimal(utemp, expo, mant))
		umax.append(todecimal(utemp1, expo, mant))

		for i in range(1,N_all-1):
			I = np.where(I1>n_all[i], 1, 0)			
			I -= np.where(I1>=n_all[i-1], thb, 0)			
			I[0] = 0
			Iset.append(I)			
			utemp = conversion(th, mant, expo)[4]
			utemp[l - n_all[i] - 1] = 0
			umin.append(todecimal(utemp1, expo, mant))
			umax.append(todecimal(utemp, expo, mant))
			
			I = np.where(I1>=n_all[i], 1, 0)
			I -= np.where(I1>=n_all[i-1], thb, 0)
			I[0] = 0
			Iset.append(I)
			utemp1[l - n_all[i] - 1] = 1
			umin.append(todecimal(utemp, expo, mant))
			umax.append(todecimal(utemp1, expo, mant))
			
		I = np.where(I1>=n_all[i+1], 1, 0)
		I -= np.where(I1>=n_all[i], thb, 0)
		I[0] = 0
		Iset.append(I)
		umin.append(todecimal(utemp1, expo, mant))
		umax.append(todecimal(thb, expo, mant))

	# Case3: u>= th
	Iset.append(thb)
	I[0] = 1
	umin.append(todecimal(thb, expo, mant))
	umax.append(todecimal(thb, expo, mant))	

	minb = np.zeros(l)
	minb[-1] = 1
	du_min = todecimal(minb, expo, mant)

	if z_all[0] != 0:
		I = np.where(I1>z_all[0],thb, 0)
		I[l - z_all[0] - 1] = 1
		I[0] = 1
		Iset.append(I)
		umin.append(umax[-1])
		utemp = np.zeros(l)
		utemp[l - z_all[0] - 1] = 1
		umax.append(todecimal(thb+utemp, expo, mant))
	else:
		utemp = np.zeros(l)
		utemp[l - z_all[0] - 1] = 1

	for i in range(1, Z_all-1):
		I = np.where(I1>z_all[i-1],thb, 0)
		I[0] = 1
		Iset.append(I)
		umin.append(umax[-1])
		umax.append(todecimal(thb+utemp, expo, mant))

		utemp[l - z_all[i] - 1] = 1
		I = np.where(I1>z_all[i],thb, 0)
		I[l - z_all[i] - 1] = 1
		I[0] = 1
		Iset.append(I)
		umin.append(umax[-1])
		utemp1 = np.zeros(l)
		utemp1[l - z_all[i] - 1] = 1
		umax.append(todecimal(thb+utemp1, expo, mant))	

	return [torch.from_numpy(np.asarray(Iset, dtype=np.single)).cuda(), 
	torch.from_numpy(np.asarray(umin, dtype=np.single)).cuda(), torch.from_numpy(np.asarray(umax, dtype=np.single)).cuda()]

def exp_lut(exp_min, exp_max):
	max_temp = np.arange(exp_min+1,exp_max+1).astype(np.single)	
	min_temp = np.arange(exp_min,exp_max).astype(np.single)
	umax = 2**(max_temp)
	umax[-1] = 1.E20
	umin = 2**(min_temp)
	umin[0] = 0
	lut = 2**(min_temp)
	lut[0] = 2**(exp_min-2)
	return torch.from_numpy(umax).cuda(), torch.from_numpy(umin).cuda(), torch.from_numpy(lut).cuda()

def grad_u_approx(Iset, umax, th, expo, mant):
	grad = []
	# Iset = Iset.cpu()
	l = expo + mant + 1
	for i in range(len(umax)):
		if umax[i] == 0:
			grad.append(0)
		else: 
			if umax[i] < th:
				grad_I = Iset[i].cpu()
				sign0 = sig(umax[i], expo, mant)
				s0 = digit2(0, l-1, mant)
				stemp = np.arange(l-1, -1, -1)
				s0 = np.where(stemp<mant, sign0/s0, s0)
				s0 = np.where((stemp>=mant) & (stemp<l-2), 1/(2**(s0)-1), s0)
				s0 = np.where(stemp == l-2, 0, s0)
				# s0 = np.where((stemp>=mant) & (stemp != l-1), 2**s0, s0)
				if umax[i] < 0:
					s0[0] = 0.5
				else:
					s0[0] = -0.5
				grad.append(np.dot(grad_I, s0)/umax[i])			
			else:
				grad_I = Iset[i].cpu()
				sign0 = sig(umax[i], expo, mant)
				s0 = digit2(0, l-1, mant)
				stemp = np.arange(l-1, -1, -1)
				s0 = np.where(stemp<mant, sign0/s0, s0)
				s0 = np.where((stemp>=mant) & (stemp<l-2), 1/(1-2**(-s0)), s0)
				s0 = np.where(stemp == l-2, 0, s0)
				s0[0] = -0.5
				grad.append(np.dot(grad_I, s0)/umax[i])

	# return torch.from_numpy(np.asarray(grad, dtype=np.single)).cuda()
	return torch.tensor(grad).cuda()
