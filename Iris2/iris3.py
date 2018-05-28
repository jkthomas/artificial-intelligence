from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydot
import io
import graphviz
import math

iris = datasets.load_iris()
x = iris.data
y = iris.target

train, test, train_targets, test_targets = train_test_split(x, y, test_size=0.50) 
#9. Skonstruować drzewa klasyfikacyjne korzystając z innych parametrów determinujących jego strukturę
#takich jak minimalna liczba próbek uczących w liściu drzewa min_samples_leaf, maksymalna dopuszczalna liczba liści drzewa (max_leaf_nodes).

clf = tree.DecisionTreeClassifier(criterion='entropy',  max_depth=2, min_samples_split=2, min_samples_leaf=2, max_leaf_nodes=10)
clf = clf.fit(x, y)
y = clf.predict(x)

score = (iris.target == y).sum()
print("Poprawnie zaklasyfikowanych. : ", score)
print("Sprawnosc: ", float(score) / len(y))

#10. Skonstruować drzewo klasyfikacyjne dla zbioru iris w podprzestrzeni cech złożonej jedynie z dwóch pierwszych atrybutów. Ocenić sprawność uzyskanego rozwiązania.

x = iris.data[:, :2]
y = iris.target

train, test, train_targets, test_targets = train_test_split(x, y, test_size=0.50)
 
clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(x, iris.target)
y = clf.predict(x)
 
proper = (iris.target == y).sum()
print("2P Poprawnie zaklas. : ", proper)
print("2P Sprawnosc: ", float(proper) / len(y))
