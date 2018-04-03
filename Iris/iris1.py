import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import matplotlib.pyplot as plt

#1 Wczytanie zbioru i podanie ilosci probek/atrybutów
data = []
with open("iris.data", 'r+') as iris:
    for line in iris:
        line = line.replace('\n','')
        row = line.split(',')
        data.append(row)

for row in data:
    for i in range(0,len(row)-1):
        row[i] = float(row[i])
      
print("Próbki:", len(data))
print("Atrybuty każdej próbki:", len(data[0]))

#2 Wartosci dla probek 10 i 75, ich odleglosc euklidesowa
sampleA = data[9].copy()
sampleB = data[74].copy()
print("Próbka 10:", sampleA)
print("Próbka 75:", sampleB)
del sampleA[-1]
del sampleB[-1]
subtractAB = np.subtract(sampleA, sampleB)
euclid = np.dot(np.transpose(subtractAB), subtractAB)
print("Odległość euklidesowa próbek 10 i 75:", euclid)

del sampleA
del sampleB

#3 Min, max, srednie, odchylenie kazdego attr
col1 = [column[0] for column in data]
col2 = [column[1] for column in data]
col3 = [column[2] for column in data]
col4 = [column[3] for column in data]
col5 = [column[4] for column in data]

print("Najmniejsze wartości:", min(col1), ",", min(col2), ",",
      min(col3), ",", min(col4))
print("Największa wartość:", max(col1), ",", max(col2), ",",
      max(col3), ",", max(col4))

print("Średnie:", round(np.mean(col1),3), ",", round(np.mean(col2),3), ",",
      round(np.mean(col3),3), ",", round(np.mean(col4),3))
print("Odchylenia:", round(np.std(col1),3), ",", round(np.std(col2),3), ",",
      round(np.std(col3),3), ",", round(np.std(col4),3))

#4 Wizualizacja atrybutów 1,2
plt.scatter(col1, col2, color=['red','blue'])
plt.show()

#5 Wizualizacja atrybutów 1,3 z podziałem na klasy
virginica1, versicolor1, setosa1 = [], [], []
virginica3, versicolor3, setosa3 = [], [], []
for i in range(0, len(col5)):
    if col5[i] == 'Iris-virginica':
        virginica1.append(col1[i])
        virginica3.append(col3[i])
    elif col5[i] == 'Iris-versicolor':
        versicolor1.append(col1[i])
        versicolor3.append(col3[i])
    elif col5[i] == 'Iris-setosa':
        setosa1.append(col1[i])
        setosa3.append(col3[i])
plt.scatter(virginica1, virginica3, color='red')
plt.scatter(versicolor1, versicolor3, color='blue')
plt.scatter(setosa1, setosa3, color='green')
plt.show()
del col1, col2, col3, col4

#6 Wartości średnie atrybutów 1,3 z podziałem na klasy
print("Atrybut 1 versicolor: ", round(np.mean(virginica1),3))
print("Atrybut 3 versicolor: ", round(np.mean(virginica3),3))
print("Atrybut 1 setosa: ", round(np.mean(setosa1),3))
print("Atrybut 3 setosa: ", round(np.mean(setosa3),3))
del virginica1, virginica3, versicolor1, versicolor3, setosa1, setosa3

#7 Normalizacja i powtórzenie kroku 3
data_norm = data.copy()
for column in data_norm:
    del column[-1]
data_norm = preprocessing.scale(data_norm)

col1 = [column[0] for column in data_norm]
col2 = [column[1] for column in data_norm]
col3 = [column[2] for column in data_norm]
col4 = [column[3] for column in data_norm]

print("Najmniejsze wartości:", min(col1), ",", min(col2), ",",
      min(col3), ",", min(col4))
print("Największa wartość:", max(col1), ",", max(col2), ",",
      max(col3), ",", max(col4))

print("Średnie:", np.mean(col1), ",", np.mean(col2), ",",
      np.mean(col3), ",", np.mean(col4))
print("Odchylenia:", round(np.std(col1),3), ",", round(np.std(col2),3), ",",
      round(np.std(col3),3), ",", round(np.std(col4),3))
del data, data_norm, col1, col2, col3, col4

#8 Generacja zbioru danych (N-ilość próbek)
N = 10
x1 = 10*np.random.rand(N)
x2 = 1*np.random.randn(N)-2
plt.scatter(x1,x2)
plt.show()

#9 Macierze odległości
data=np.vstack((x1,x2))
data=data.conj().transpose()
euclid = metrics.pairwise.pairwise_distances(data, metric='euclidean')
mahal = metrics.pairwise.pairwise_distances(data, metric='mahalanobis')
minkow = metrics.pairwise.pairwise_distances(data, metric='minkowski')
print("Odległość euklidesowa: ", euclid)
print("Odległość mahalanobisa: ", mahal)
print("Odległość Minkowskiego: ", minkow)

#10 Skalowanie liniowe i ponowne liczenie odległości
data_norm = preprocessing.scale(data)
euclid = metrics.pairwise.pairwise_distances(data, metric='euclidean')
mahal = metrics.pairwise.pairwise_distances(data, metric='mahalanobis')
minkow = metrics.pairwise.pairwise_distances(data, metric='minkowski')
print("Znormalizowana odległość euklidesowa: ", euclid)
print("Znormalizowana odległość mahalanobisa: ", mahal)
print("Znormalizowana odległość Minkowskiego: ", minkow)
