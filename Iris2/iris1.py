from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydot
import io
import graphviz
import math

#W tym przypadku pobieramy dataset, zamiast korzystac z wlasnego pliku (dataset zawiera 150 probek)
#1. Wczytać zbiór uczący iris i dokonać jego podziału na część uczącą i testową (po 75 próbek dla uczenia i testowania)

iris = datasets.load_iris()
x = iris.data
y = iris.target

train, test, train_targets, test_targets = train_test_split(x, y, test_size=0.50) 

#2. Skonstruować drzewo klasyfikacyjne dla domyślnych wartości parametrów na podstawie zbioru uczącego i dokonać jego wizualizacji.

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
y = clf.predict(iris.data)
out = io.StringIO()
tree.export_graphviz(clf, out_file=out)
graph = pydot.graph_from_dot_data(out.getvalue())
graph[0].write_png("visualization2.png")

#3. Ocenić uzyskaną sprawność klasyfikacji na zbiorze testowym. Ile elementów zostało niepoprawnie zaklasyfikowanych?

clf = neighbors.KNeighborsClassifier(k, weights=’uniform’, metric=’euclidean’)
clf.fit(train, train_targets)
clfScore = clf.score(train, test_targets)
print("Ilość niepoprawnie zaklasyfikowanych elementów: ", math.ceil(len(train) * (1 - clfScore)))

#4. Odczytać wartości parametrów drzewa klasyfikacyjnego. Jakie kryterium decyduje o wyborze testu dla wartości atrybutów?

print("Odp. na pyt. 4: Decyduje indeks Giniego")

#5. Przetestować działanie algorytmu drzewa klasyfikacyjnego na próbkach zbioru testowego i ocenić sprawność klasyfikacji.

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
y = clf.predict(iris.data)
 
proper = (iris.target == y).sum()
print("Sprawnosc klasyfikacji: ", float(proper) / len(y))
