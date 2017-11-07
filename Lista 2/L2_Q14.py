def mahal(x, u, cov):
	# Compute mahalanobis distance between x, u with covariance cov
	b = (x.T - u).T
	A = np.linalg.inv(cov)

	# print("\nb shape: ", b.shape)
	# print("\nb.T shape: ", b.T.shape)
	# print("\nA shape: ", A.shape )

	# result = np.sqrt(np.absolute(np.dot(b.T, np.dot(A, b))))
	result = (b.T.dot(A)*b.T)
	return result.sum(axis=1)

import numpy	 				as np
from scipy.stats				import multivariate_normal
from matplotlib import pyplot 	as plt

N1 = int(np.floor(1000/3))
N = 3*N1

## Q13a
# print("\nLetra a")
# u1  = np.array([1,1]).T
# u2  = np.array([12, 8]).T
# u3  = np.array([16, 1]).T
# cov1 = cov2 = cov3 = np.identity(2)

## Q12b
# print("\nLetra b")
# u1  = np.array([1,1]).T
# u2  = np.array([14, 7]).T
# u3  = np.array([16, 1]).T
# cov1 = cov2 = cov3 = np.array([[5, 3],[3, 4]])

## Q12c
print("\nLetra c")
u1  = np.array([1,1]).T
u2  = np.array([10, 5]).T
u3  = np.array([11, 1]).T
cov1 = cov2 = cov3 = np.array([[7, 4],[4, 5]])

# Create random data
x1 = np.empty([N1, 2])
x2 = np.empty([N1, 2])
x3 = np.empty([N1, 2])
for i in range(N1):
	x1[i,:] = multivariate_normal.rvs(mean=u1, cov=cov1)
	x2[i,:] = multivariate_normal.rvs(mean=u2, cov=cov2)
	x3[i,:] = multivariate_normal.rvs(mean=u3, cov=cov3)

x = np.reshape([x1, x2, x3], [N,2])
y = np.reshape([np.zeros([N1,1]), np.ones([N1,1]), np.ones([N1,1])*2], [N,]).astype(int)

plt.scatter(x[:,0], x[:, 1])
plt.show()

## Euclidean classifier
# Class 1: y = 0
# Class 2: y = 1
# Class 3: y = 2
euclid1 = np.linalg.norm(x - u1, 2, 1)
euclid2 = np.linalg.norm(x - u2, 2, 1)
euclid3 = np.linalg.norm(x - u3, 2, 1)

estEuclid = np.empty([N,])
estEuclid = np.argmin([euclid1, euclid2, euclid3], 0)
# print("shape: ", np.shape([euclid1, euclid2, euclid3]))

errEuclid = np.mean(np.where(estEuclid == y, 0, 1))
print("\nError euclid: ", errEuclid)

## Mahalanobis classifier
mahal1 = mahal(x.T, u1.T, cov1)
mahal2 = mahal(x.T, u2.T, cov2)
mahal3 = mahal(x.T, u3.T, cov3)

# print("\nmahal1: \n", mahal1)

estMahal = np.empty([N,])
estMahal  = np.argmin([mahal1, mahal2, mahal3], 0)

errMahal = np.mean(np.where(estMahal == y, 0, 1))
print("Error mahal : ", errMahal)