"""
PCA_2dim.py

"""

import os
import pandas as pd
from sklearn.decomposition import PCA

import pickle

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_path = os.path.join(path, file)

new_file = "datos_PCA_2d.csv"
new_full_path = os.path.join(path, new_file)

df = pd.read_csv(full_path, sep=';')


datos = df.iloc[:, 4:].values

pca = PCA(n_components=2)
pca.fit(datos)

arch = 'PCA2d_model.pkl'
arch = os.path.join(path, arch)

with open(arch, 'wb') as f:
    pickle.dump(pca, f)


"""
datos_PCA = pca.transform(datos)

df2 = pd.DataFrame(datos_PCA, columns=['eje_x', 'eje_y'])

df1 = df.iloc[:,0:4]

df = pd.concat([df1, df2], axis=1)

df.to_csv(new_full_path, sep=';', index=False)

print(df.head())
"""

