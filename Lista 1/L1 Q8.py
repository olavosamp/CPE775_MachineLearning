# Lista 1 
# Q8

import numpy as np
from scipy import stats

K = 1000

D 	   = np.zeros([2, 2*K])
D_test = np.zeros([2, K])

## Training set
# D[0,:] are input values
# D[1,:] are output values, y = x^2 
D[0,:] = np.random.uniform(-1.0, 1.0, 2*K)
D[1,:] = np.power(D[0,:], 2)

## Test set
D_test[0,:] = np.random.uniform(-1.0, 1.0, K)
D_test[1,:] = np.power(D_test[0,:], 2)

print("Datasets D")
print(D)

## Linear hypotheses
# h[0,:] are zero order coeficients
# h[1,:] are first order coeficients
h = np.zeros([2,K])

# h[1,:] = (D[1, 1::2] - D[1, 0::2])/(D[0, 1::2] - D[0, 0::2])
# h[0,:] = D[1, 0::2] - h[1,:]*D[0, 0::2]

# Finding coefficients

for i in range(0, 2*K, 2):
	# x = [D[0, i], D[0, i+1]]
	# y = [D[1, i], D[1, i+1]]
	x = np.array([D[0, i], D[0, i+1]])
	y = np.array([D[1, i], D[1, i+1]])
	#h[:,int(i/2)] = np.polyfit(x, y, 1)
	slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
	h[0,int(i/2)] = intercept
	h[1,int(i/2)] = slope

print("\nh(x) = ax + b")
print(h)

g = np.zeros([2,K])

print("Shape h", np.shape(h[1,:]))
print("Shape D slice", np.shape(D[0,0::2]))

g[0,:] = h[0,:] + np.multiply(h[1,:],D[0,0::2])
g[1,:] = h[0,:] + np.multiply(h[1,:],D[0,1::2])
g = np.reshape(g,[np.size(g)])
print("Shape g", np.shape(g))

# f = np.zeros([2,K])
# f[0,:] = np.power(D[0,0::2],2)
# f[1,:] = np.power(D[0,1::2],2)
# f = np.sum(f,0)
f = D[1,:]


gAvg = np.mean(g)
print("\nAverage g(x):")
print(gAvg)

# Bias
print("\nBias")
bias = np.power(gAvg - np.mean(f), 2)
bias = np.mean(bias)
print(bias)

Davg = np.mean(D[0,:])
print("\nD avg: ", Davg)

# Variance
print("\nVariance")
var = np.var(g)
print("Var: ", var)

# Population error
eOut = np.mean(np.power(g[:] - f[:], 2))
print("\nEout: ", eOut)
print("Eout = bias + var: ", bias+var)