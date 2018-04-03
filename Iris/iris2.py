import numpy as np
import matplotlib.pyplot as plt
#11. Funkcje klasyfikujące

#x - wektor, na podstawie którego określa się klasę
#x[0] = x, x[1] = y
#gA, A - numer ćwiartki, do której należą elementy klasy
def g2(x):
    return -x[0]+x[1]

def g4(x):
    return x[0]-x[1]

def classify(x):
    if g2(x)>g4(x):
        return 1
    else:
        return 2;

#Powierzchnia decyzyjna:
    #g2=g4
    #-x[0]+x[1]=x[0]-x[1]
    #x[0]=x[1] - prosta (x=y)

#12. Test klasyfikatora
N = 20
class1 = 2*np.random.rand(N,2)
class2 = -2*np.random.rand(N,2)
data=np.vstack((class1,class2))

y1=[g2(data[i,:]) for i in range(2*N)]
y2=[g4(data[i,:]) for i in range(2*N)]
y1=np.round(y1,2)
y2=np.round(y2,2)
print('Wartości funkcji klasyfikujacych')
print(y1)
print(y2)

labels=np.array([classify(data[i,:]) for i in range(2*N)])
print('Decyzje klasyfikatora:')
print(labels)

c1 = (labels==1).nonzero()
c2 = (labels==2).nonzero()

plt.ylim(ymax = 2, ymin = -2)
plt.xlim(xmax = 2, xmin = -2)

plt.scatter(data[c1,0],data[c1,1],color='blue')
plt.scatter(data[c2,0],data[c2,1],color='red')

plt.plot([2, -2],[2, -2])
plt.show()
