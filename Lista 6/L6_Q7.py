import numpy 				as np

from scipy.stats import multivariate_normal

def fisherDisc(dataA, dataB):
	# Compute Fisher Discriminant between classes A and B
	uA = np.mean(dataA)
	uB = np.mean(dataB)
	varB = np.var(dataA)
	varA = np.var(dataB)

	fdr = np.sum((uA - uB)/(np.power(varA, 2) - np.power(varB, 2)))
	return fdr

M = 100				# Number of data points points per class
meanA = [2, 4]		# Mean of class A
meanB = [2.5, 10]	# Mean of class B

# cov = np.eye(2)		# Letra A
cov = 0.25*np.eye(2)# Letra B

# Sample data points from multivariate Gaussian
xA = np.zeros((M, 2))
xB = np.zeros((M, 2))
for i in range(M):
	xA[i,:] = multivariate_normal.rvs(mean=meanA, cov=cov)
	xB[i,:] = multivariate_normal.rvs(mean=meanB, cov=cov)

#
print("")
print('Class A: \n\tMean: {}\n\tCov: {}'.format(np.mean(xA, 0), np.var(xA, 0)))
print('Class B: \n\tMean: {}\n\tCov: {}'.format(np.mean(xB, 0), np.var(xB, 0)))

fdr1 = fisherDisc(xA[:,0], xB[:,0])
fdr2 = fisherDisc(xA[:,1], xB[:,1])
print('\nFisher discriminant (A, B): \n\tAtrib 1: {:2.3f}\n\tAtrib 2: {:2.3f}'.format(fdr1, fdr2))
