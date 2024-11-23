"""
PCA_100dim.py

"""

import os
import pandas as pd
from sklearn.decomposition import PCA

import pickle

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_path = os.path.join(path, file)

new_file = "datos_PCA_100d.csv"
new_full_path = os.path.join(path, new_file)

df = pd.read_csv(full_path, sep=';')


datos = df.iloc[:, 4:].values

pca = PCA(n_components=100)
pca.fit(datos)


# Guardar el modelo
file_model = 'pca_model.pkl'
full_file_model = os.path.join(path, file_model)
with open(full_file_model, 'wb') as f:
    pickle.dump(pca, f)



datos_PCA = pca.transform(datos)

df2 = pd.DataFrame(datos_PCA, columns=[i for i in range(100)])

df1 = df.iloc[:,0:4]

df = pd.concat([df1, df2], axis=1)

df.to_csv(new_full_path, sep=';', index=False)


"""
file_model = 'pca_model.pkl'
full_file_model = os.path.join(path, file_model)
with open(full_file_model, 'rb') as f:
    pca = pickle.load(f)
"""