import numpy as np
import sklearn.model_selection as skms
from scipy import stats
np.random.seed(0)
# N=1000
nDatasets=1000
datasetSize = 2
limit = 1
inputX = np.random.uniform(-limit,limit,(nDatasets,datasetSize))

def g(a,b):
	def g(x):
		return a*x+b
	return g

def MSE(x,y):
	return np.mean((x-y)**2)
print (MSE(np.array([1,2]),np.array([5,6])))

# dataset = np.vstack((inputX,inputX**2))
y = inputX**2

print (inputX.shape)
print (y.shape)
# X_train, X_test, y_train, y_test = skms.train_test_split(
    # inputX, y, test_size=0.1, random_state=42)

slopes = np.zeros((len(inputX),))
intercepts = np.zeros((len(inputX),))

print (slopes.shape)
for index in range(len(inputX)):
	slope, intercept, r_value, p_value, std_err = stats.linregress(inputX[index], y[index])
	slopes[index] = slope
	intercepts[index] = intercept
slopeAvg = slopes.mean()
interceptAvg = intercepts.mean()

nTestSamps = 1000
xTest = np.random.uniform(-limit,limit,nTestSamps)
yTest = xTest**2

print (slopeAvg)
print (interceptAvg)

gAvg = g(slopeAvg,interceptAvg)
bias = MSE(gAvg(xTest),yTest)
print ("bias:",bias)
#gAvg(10)

var=0
eout = 0 
for index in range(len(slopes)):
	gD = g(slopes[index],intercepts[index])
	var+=MSE(gD(xTest),gAvg(xTest))
	eout+=MSE(gD(xTest),y)
var/=len(slopes)
eout/=len(slopes)

print ("var:",var)
print ("eout:",eout)




print (inputX[0:4])
print (y[0:4])

