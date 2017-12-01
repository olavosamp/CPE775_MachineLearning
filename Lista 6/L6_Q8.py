import numpy 				as np

from scipy.stats import multivariate_normal

def wScatter(x, N):
    # Compute Within-Class Scatter Matrix for one class
    # x represents data points from a single class'
    # N is the total number of data points from every class
    probX = len(x)/N
    cov = np.var(x)
    return probX*cov

def bScatter(x, N, u):
    # Compute Between-Class Scatter Matrix for one class
    # x represents data points from a single class'
    # N is the total number of data points from every class
    # u is the mean of all data points
    meanX = np.mean(x, 0)
    probX = len(x)/N
    # print("meanX: ", meanX)
    # print("lenX: ", len(x))
    # print("probX: ", probX)
    # print("u: ", u)
    # print("meanx - u: ", meanX - u)
    # print("")

    result = probX*(meanX - u)*(meanX - u).T

    return result

M = 100				# Number of data points per class

# # Letra A
# meanA = [-10,-10]   # Mean of classes A through D
# meanB = [-10, 10]   #
# meanC = [ 10,-10]   #
# meanD = [ 10, 10]   #
# cov = np.eye(2)

# # Letra C
# meanA = [-1,-1]   # Mean of classes A through D
# meanB = [-1, 1]   #
# meanC = [ 1,-1]   #
# meanD = [ 1, 1]   #
# cov = np.eye(2)

# Letra D
meanA = [-10,-10]   # Mean of classes A through D
meanB = [-10, 10]   #
meanC = [ 10,-10]   #
meanD = [ 10, 10]   #
cov = 3*np.eye(2)


# Sample data points from multivariate Gaussian
xA = np.zeros((M, 2))
xB = np.zeros((M, 2))
xC = np.zeros((M, 2))
xD = np.zeros((M, 2))
for i in range(M):
    xA[i,:] = multivariate_normal.rvs(mean=meanA, cov=cov)
    xB[i,:] = multivariate_normal.rvs(mean=meanB, cov=cov)
    xC[i,:] = multivariate_normal.rvs(mean=meanC, cov=cov)
    xD[i,:] = multivariate_normal.rvs(mean=meanD, cov=cov)

# Complete dataset
x = np.reshape(np.vstack((xA, xB, xC, xD)), (4*M, 2))

y = np.empty(4*M)
y[0:M]      = 0
y[M:2*M]    = 1
y[2*M:3*M]  = 2
y[3*M:]     = 3

# Compute Within-Class Scatter matrix for the dataset
withinScatter = np.zeros((2,2))
betweenScatter = np.zeros((2,2))
for xAux in [xA, xB, xC, xD]:
    withinScatter = withinScatter + wScatter(xAux, 4*M)
    betweenScatter = betweenScatter + bScatter(xAux, 4*M, np.mean(x, 0))

print("")
print('Class A: \n\tCount: {}\n\tMean: {}\n\tCov: {}'.format(np.sum(np.where(y == 0, 1, 0)), np.mean(xA, 0), np.var(xA, 0)))
print('Class B: \n\tCount: {}\n\tMean: {}\n\tCov: {}'.format(np.sum(np.where(y == 1, 1, 0)), np.mean(xB, 0), np.var(xB, 0)))
print('Class C: \n\tCount: {}\n\tMean: {}\n\tCov: {}'.format(np.sum(np.where(y == 2, 1, 0)), np.mean(xC, 0), np.var(xC, 0)))
print('Class D: \n\tCount: {}\n\tMean: {}\n\tCov: {}'.format(np.sum(np.where(y == 3, 1, 0)), np.mean(xD, 0), np.var(xD, 0)))

print("\nWithin-Class Scatter Matrix: \n{}".format(withinScatter))
print("\nBetween-Class Scatter Matrix: \n{}".format(betweenScatter))

mixtureScatter = withinScatter + betweenScatter
print("\nMixture Scatter Matrix: \n{}".format(mixtureScatter))

# Letra B
# Tr{Inv(Sw)*St}
# measureJ = np.linalg.inv(withinScatter)*betweenScatter
measureJ = (np.linalg.pinv(withinScatter)*betweenScatter).trace()
print("\nJ = Tr(Inv(Sw)*St): \n{}".format(measureJ))
