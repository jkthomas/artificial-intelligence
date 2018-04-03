import scipy.io as sio
import numpy as np
import math
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import neighbors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#1
banana = sio.loadmat("banana.mat")

train_data = banana['train_data']
train_labels = banana['train_labels']
train_labels = np.array(train_labels)
test_data = banana['test_data']
test_labels = banana['test_labels']
test_labels = np.array(test_labels)
del banana

train, ignore, train_targets, ignore = train_test_split(train_data, train_labels.ravel(), test_size=0.70)
ignore, test, ignore, test_targets = train_test_split(test_data, test_labels.ravel(), test_size=0.70)

#2
gaussiannb = GaussianNB()
clf = gaussiannb.fit(train, train_targets)
Z = clf.predict(test)

#3
c1 = (Z == 1).nonzero()
c2 = (Z == 2).nonzero()
C = 1.0
h = .02
x_min, x_max = test[:, 0].min() - 1, test[:, 0].max() + 1
y_min, y_max = test[:, 1].min() - 1, test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
print("Sprawność klasyfikatora: %f" % clf.score(test,test_targets))
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap="RdYlGn")
plt.scatter(test[c1, 0], test[c1, 1], c="r", label="Grupa 1")
plt.scatter(test[c2, 0], test[c2, 1], c="g", label="Grupa 2")
plt.legend()
plt.show()

#4
print(round(clf.score(test, test_targets) * 100, 2))

#5
classifier = NearestCentroid()

#6
classifier.fit(train, train_targets)
Z = classifier.predict(test)

#7
c1 = (Z == 1).nonzero()
c2 = (Z == 2).nonzero()
plt.scatter(test[c1, 0], test[c1, 1], c="g", label="Klasa 1")
plt.scatter(test[c2, 0], test[c2, 1], c="r", label="Klasa 2")
plt.legend()
plt.scatter(classifier.centroids_[:, 0], classifier.centroids_[:, 1], c="b")
plt.show()

#8
print("Sprawnosc klasyfikatora: %f" % classifier.score(test,test_targets))

#9
best_core = 0
best_k = 0
for k in range(1, 10):
    clf = neighbors.KNeighborsClassifier(k, weights='uniform', metric='euclidean')
    clf.fit(train, train_targets)
    score = clf.score(test, test_targets)
    if score > best_core:
        best_core = score
        best_k = k
print("Najwyższa sprawność: ", best_core, ". Osiągnięta dla k = ", best_k)

#10
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
Z = neighbors.KNeighborsClassifier(best_k, weights='uniform',
     metric='euclidean').fit(train, train_targets).predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure()
#plt.pcolormesh(xx, yy, Z)
plt.pcolormesh(xx, yy, Z, cmap="RdYlGn")
plt.scatter(test[:, 0], test[:, 1], c=test_targets, cmap="binary")
#plt.scatter(test[c1, 0], test[c1, 1], c="g", label="Klasa 1")
#plt.scatter(test[c2, 0], test[c2, 1], c="r", label="Klasa 2")
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()

#11
classifier = neighbors.KNeighborsClassifier(best_k, weights='uniform', metric='euclidean')
classifier.fit(train, train_targets)
score = classifier.score(test, test_targets)
print("Ilość źle sklasyfikowanych obiektów: ", math.floor(len(test_data) * (1 - score)))
