"""
tSNE_2dim.py

"""

import os
import pandas as pd
from sklearn.manifold import TSNE

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_path = os.path.join(path, file)

new_file = "datos_tSNE-2d.csv"
new_full_path = os.path.join(path, new_file)

df = pd.read_csv(full_path, sep=';')

datos = df.iloc[:, 4:].values

tsne = TSNE(n_components=2, perplexity=7, n_iter=2350, random_state=42)

datos_tSNE = tsne.fit_transform(datos)

df1 = df.iloc[:, 0:4]

df2 = pd.DataFrame(datos_tSNE, columns=['eje_x', 'eje_y'])

df = pd.concat([df1, df2], axis=1)

df.to_csv(new_full_path, sep=';', index=False)

print(df.head())