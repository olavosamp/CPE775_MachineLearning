#
# Lista 2
# Q12
# Olavo Sampaio
#
from scipy.stats import multivariate_normal
import numpy 			as np

N = 100;

u1  = np.array([1,1])
u2  = np.array([1.5, 1.5])
cov = np.array([[1.01, 0.2],[0.2, 1.01]])

# x = np.transpose(np.array([np.linspace(1.4, 1.6, N), np.linspace(1.4, 1.6, N)]))
x1 = np.empty([N, 2])
x2 = np.empty([N, 2])
for i in range(N):
	x1[i,:] = multivariate_normal.rvs(mean=u1, cov=cov)
	x2[i,:] = multivariate_normal.rvs(mean=u2, cov=cov)

print("\nShape x1: ", np.shape(x1))
print("\nShape x2: ", np.shape(x2))

x = np.reshape([x1, x2], [2*N,2])
y = np.reshape([np.zeros([N,1]), np.ones([N,1])], [2*N,])

print("\nShape x: ", np.shape(x))
print("\nShape y: ", np.shape(y))
# print("\nShape u1: ", np.shape(u1))
# print("\nShape u2: ", np.shape(u2))
# print()

prob1 = multivariate_normal.pdf(x, mean=u1, cov=cov)
prob2 = multivariate_normal.pdf(x, mean=u2, cov=cov)

## Q12a - Min prob de Erro
# Choose class 1 (y=0) or class 2 (y=1)
est = np.where(prob1 >= prob2, 0, 1)

error = np.mean(np.where(est == y, 0, 1))
print("\nShape est: ", np.shape(est))
print("\nShape error: ", np.shape(error))

# print("\nest = ", est)
print("\nerror = ", error)

## Q12b - Min Risco
risk12 = 1
risk21 = 0.5

estRisk = np.where(risk12*prob1 >= risk21*prob2, 0, 1)
errorRisk = np.mean(np.where(estRisk == y, 0, 1))

print("\nerror risk= ", errorRisk)