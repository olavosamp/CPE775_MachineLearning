import time
import numpy 	as np
import pandas 	as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

path = r'F:\\Program Files\\Arquivos Incomuns\\Relevante\\UFRJ\\Aprendizado de Maquina - CPE775\\Listas\\CPE775_MachineLearning\\Dataset\\optdigits.tra.csv'

data = pd.read_csv(path)

N = data.shape[1] - 1	# Atributes

x = data.iloc[:,:N]		# Drop labels

print(x.shape)

mse = np.zeros(N)
for component in range(1, N, 1):
	pca = PCA(component)
	x_red = pca.fit_transform(x)
	x_inv = pca.inverse_transform(x_red)
	
	diff = x - x_inv

	mse[component-1] = np.sum(np.sum(np.power(diff, 2)))
	# print("{}: {}".format(component, mse[component-1]))

plt.figure()
plt.plot(range(1,N+1), mse, 'k.')
plt.title("PCA Reconstruction Error")
plt.xlabel("Nº of Components Kept")
plt.ylabel("MSE")
plt.show()