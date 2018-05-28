from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree
import pydot
import io
import graphviz
import math

iris = datasets.load_iris()
x = iris.data

train, test, train_targets, test_targets = train_test_split(x, iris.target, test_size=0.50) 
#6. Skonstruować i wyświetlić drzewo ponownie ograniczając jego głębokość do dwóch oraz trzech. Jaką w tym przypadku osiągamy sprawność klasyfikacji?

clf = tree.DecisionTreeClassifier(max_depth=2)
clf = clf.fit(x, iris.target)
y = clf.predict(x)
proper = (iris.target == y).sum()
print("Sprawnosc klasyfikacji: ", float(proper) / len(y))
out = io.StringIO()
tree.export_graphviz(clf, out_file=out)
graph = pydot.graph_from_dot_data(out.getvalue())
graph[0].write_png("visualization6_2.png")

clf = tree.DecisionTreeClassifier(max_depth=3)
clf = clf.fit(x, iris.target)
y = clf.predict(x)
proper = (iris.target == y).sum()
print("Sprawnosc klasyfikacji: ", float(proper) / len(y))
out = io.StringIO()
tree.export_graphviz(clf, out_file=out)
graph = pydot.graph_from_dot_data(out.getvalue())
graph[0].write_png("visualization6_3.png")

#7. Skonstruować drzewo klasyfikacyjne korzystając z kryterium przyrostu informacji dla wyboru testu (Wskazówka: criterion=’entropy’).

clf = tree.DecisionTreeClassifier(criterion=’entropy’)
clf = clf.fit(x, iris.target)
y = clf.predict(x)
proper_new = (iris.target == y).sum()

#8. Przetestować uzyskane drzewo na zbiorze testowym i porównać wynik z drzewem uzyskanym dla indeksu Giniego.

print("Sprawnosc dla indeksu Giniego: ", float(proper) / len(y))
print("Sprawnosc dla kryterium przyrostu: ", float(proper_new) / len(y))

