import numpy 				as np
import matplotlib.pyplot    as plt
from scipy.stats            import multivariate_normal
from sklearn.cluster        import KMeans


M = 100

meanA = np.transpose([2, 2])   # Mean of classes A through D
meanB = np.transpose([6, 6])   #
meanC = np.transpose([10,2])   #
cov = 0.5*np.eye(2)

xA = np.zeros((2, M))
xB = np.zeros((2, M))
xC = np.zeros((2, M))
for i in range(M):
    xA[:, i] = multivariate_normal.rvs(mean=meanA, cov=cov)
    xB[:, i] = multivariate_normal.rvs(mean=meanB, cov=cov)
    xC[:, i] = multivariate_normal.rvs(mean=meanC, cov=cov)

X = np.concatenate([xA, xB, xC], 1).T

# print(xA.shape)
# print(xB.shape)
# print(xC.shape)
# print(X.shape)

clusters = range(2, 5)
for nClusters in clusters:
    kmeans = KMeans(nClusters, init='random', n_init=10, algorithm='full')
    kmeans.fit_predict(X)

    # Plot the decision boundary. For that, we will assign a color to each
    h = 0.01    # meshgrid granularity
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=120, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
