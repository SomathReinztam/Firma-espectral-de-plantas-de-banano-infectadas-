"""
PCA_2d.py

"""
import os
import pandas as pd
from sklearn.decomposition import PCA
import pickle

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

datos = df.iloc[:, 4:].values

pca = PCA(n_components=2)
pca.fit(datos)

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code4\Sets"
path = path.replace('\\', '/')

file = 'model_PCA_2d.pkl'
file = os.path.join(path, file)

with open(file, 'wb') as f:
    pickle.dump(pca, f)

