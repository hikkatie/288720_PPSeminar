# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import mdshare
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

#ustawienie parsera
parser = argparse.ArgumentParser(description = "Zadanie - rzutowanie tensora na przestrzeń trójwymiarową")

#dodanie argumentow 
parser.add_argument('-d' , '--dimensions', type = int, default = 3, help = " Liczna wymiarow, do ktorych nastepuje zredukowaniu; domyslnie: wszystkie wymiary wejsciowe zachowane ")
parser.add_argument('-s' , '--step', type = int, default = 500, help = " Liczba próbkowania - zmniejszenie liczby danych na wejsciu; domyslnie: 500 ")
parser.add_argument('-sol', '--solver', type = str, default = 'auto', help = "Metoda rozwiazania oparta na SVD: 'auto' - najbardziej wydajna metoda losowa, inne: 'full', 'arpack','randomized'; domyslnie: auto " )

#przypisanie odpowiednich argumentow
args = parser.parse_args()
Dimensions = args.dimensions
Step = args.step
Solver = args.solver

#pobieranie danych z biblioteki mdshare
dataset = mdshare.fetch('alanine-dipeptide-3x250ns-heavy-atom-distances.npz')
with np.load(dataset) as f:
    X = np.vstack([f[key] for key in sorted(f.keys())])

#Standaryzacja danych wejsciowych w celu ujednolicenia danych (można pominąc)
X_std = StandardScaler().fit_transform(X)

if Dimensions == 3:
    
    #wlasciwa redukcja wymiarow
    Y = PCA(n_components = Dimensions, svd_solver = Solver).fit_transform(X_std[::Step])
    
    #dopasowanie do zakresu od -pi do +pi
    Y[:, 0] = np.interp(Y[:, 0], (Y[:, 0].min(), Y[:, 0].max()), (-np.pi, np.pi)) 
    Y[:, 1] = np.interp(Y[:, 1], (Y[:, 1].min(), Y[:, 1].max()), (-np.pi, np.pi)) 
    Y[:, 2] = np.interp(Y[:, 2], (Y[:, 2].min(), Y[:, 2].max()), (-np.pi, np.pi)) 

    #tworzenie wykresu, os x - PC1, os y - PC2 wraz z dopasowaniem osi do zadeklarowanego zakresu, trzeci wymiar jako kolor
    plt.scatter(Y[:, 0], Y[:, 1], c=Y[:, 2], s = 15, alpha = 0.5, cmap = 'viridis', ) 
    plt.xlim(-np.pi, np.pi) 
    plt.ylim(-np.pi, np.pi) 
    plt.xticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) 
    plt.yticks([-np.pi, 0, np.pi], ['-π', 0, 'π']) 
    plt.axis('scaled') #skalowanie wykresu do zadanego pola kreslenia
    cbar=plt.colorbar() 
    cbar.set_ticks([-np.pi, 0, np.pi]) 
    cbar.set_ticklabels(['-π', 0, 'π']) 
    plt.show()

else:
    print (" Nie można przeprowadzic redukcji wymiarow  ")
    
# w celu zaprezentowania wpływu poszczegolnych skladnikow (Principal Components - PC)
# obliczono macierz kowariancji, na jej podstawie wekory i wartosci wlasne macierzy oraz wyjasnione wariancje zmiennych

#macierz kowariancji - uzyskany kszatłt [45,45]
np.cov(X_std.T)

# wektory wlasne (eig_vecs) i wartosci wlasne (eig_vals)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

#wykres procentowego udzialu wariancji w kazdym komponencie - eliminacja mniej znaczacych PC, tzw scree plot

tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
bars = ['%s' %i for i in range(1,46)]
y_pos = np.arange(len(var_exp))

plt.figure(figsize=(12,6))
plt.bar(y_pos, var_exp, color = 'darkolivegreen') 
plt.xticks(y_pos, bars)
plt.xlabel('Skladniki glowne (PC)', fontsize=12, color='black') 
plt.ylabel('Procentowa wyjasniona wariancja', fontsize=12, color='black') 
plt.title('Wariancja wyjasniona dla roznych skladnikow glownych (Principal Components)', fontsize=16, color='black') 
plt.show()

print ("Zadanie wykonane ")



