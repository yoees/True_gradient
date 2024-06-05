import numpy as np 
from fxpmath import Fxp
import matplotlib.pyplot as plt

def conversion(x, word, frac):
	y_temp = Fxp(x, signed=False, n_word=word, n_frac=frac).bin()
	y_temp = np.array(list(map(int, y_temp)))
	y_zero = np.nonzero(y_temp==0)[0]
	y_nonzero = np.nonzero(y_temp)[0]	
	y_1 = word - y_nonzero - 1
	y_2 = np.sort(word - y_zero - 1)
	nz = len(y_1)
	zlen = len(y_2)
	print(y_temp, sum(y_temp))
	return y_1, nz, y_2, zlen

def conversion_inv(x, frac):	
	y_temp = 2**(x.astype(float)-frac)
	if type(x) == np.int64:
		y = y_temp
	else:
		y = sum(y_temp)
	return y

def conversion_inv2(x, frac):	
	y_temp = 2**(-x.astype(float)+frac)
	if type(x) == np.int64:
		y = y_temp
	else:
		y = sum(y_temp)
	return y

def comp(m, z, i):
	temp = m[(m-z[i])>0]	
	return temp

word, frac = 15, 15
th = 0.295 # {0.125, 0.15625, 0.2, 0.295}
m, nz, z, zlen = conversion(th, word, frac)

u = []
grad = []

l1 = frac 
l2 = word-frac 
l = word + 1

# case 1: u < 0
u.append(th-2**(l-l1-1))
grad.append(0)

u.append(0)
grad.append(abs(-2**(l1-l+1)))

# case 2: 0<= u < theta 
if nz == 1:
	u.append(th)
	a = 2**(l1-m[0]) - 2**(l1-l+2)
	grad.append(a)
elif nz == 2:
	u.append(th - conversion_inv(m[0], frac))
	u.append(conversion_inv(m[0], frac))
	u.append(th)

	a = 2**(l1-m[0]) - 2**(l1-l+2)
	grad.append(a)
	a = 2**(l1-m[0]+1) - 2**(l1-l+2)
	grad.append(a)
	a = 2**(l1-m[1]+1) - 2**(l1-l+2) - conversion_inv2(m[0], frac)
	grad.append(a)
else:
	u.append(th - conversion_inv(m[0], frac))
	u.append(conversion_inv(m[0], frac))

	a = 2**(l1-m[0]) - 2**(l1-l+2)
	grad.append(a)
	a = 2**(l1-m[0]+1) - 2**(l1-l+2)
	grad.append(a)

	for i in range(1,nz-1):
		u.append(th - conversion_inv(m[i], frac))
		u.append(conversion_inv(m[0:i+1], frac))
		
		a = 2**(l1-m[i]) - 2**(l1-l+2) - conversion_inv2(m[0:i], frac)
		grad.append(a)
		a = 2**(l1-m[i]+1) - 2**(l1-l+2) - conversion_inv2(m[0:i], frac)
		grad.append(a)
	u.append(th)

	a = 2**(l1-m[i+1]+1) - 2**(l1-l+2) - conversion_inv2(m[0:i+1], frac)
	grad.append(a)

# case 3: u >= theta
u.append(th)
a = conversion_inv2(m, frac) - 2**(l1-l+1)
grad.append(a)

if z[0] != 0:
	u.append(th + conversion_inv(z[0], frac))
	m_cond = comp(m,z,0)
	a = conversion_inv2(m_cond, frac) + 2**(l1-z[0]) - 2**(l1-l+1)
	grad.append(a)

for i in range(zlen-1):
	u.append(th + conversion_inv(z[0:i+1], frac))
	m_cond = comp(m,z,i)
	a = conversion_inv2(m_cond, frac) - 2**(l1-l+1)
	# print(m_cond, conversion_inv2(m_cond, frac) - 2**(l1-l+1))
	grad.append(a)

	u.append(th + conversion_inv(z[i+1], frac))
	m_cond = comp(m,z,i+1)	
	a = conversion_inv2(m_cond, frac) + 2**(float(l1-z[i+1])) - 2**(l1-l+1)
	grad.append(a) 

print(u)
print(grad)
plt.xlim([-0.2, 0.75])
plt.step(u, grad, 'grey', label='pre (default)')
plt.semilogy(u, grad, 'C0o', alpha=0.5)
plt.vlines(th,0,1E4, colors='k')
plt.grid()
plt.show()

area = 0
for i in range(1,len(u)):
	area += grad[i]*(u[i] - u[i-1])
print(area)