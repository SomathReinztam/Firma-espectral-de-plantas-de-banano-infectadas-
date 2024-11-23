"""
1_Heatmap.py

"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos_no_control.csv'
file = os.path.join(path, file)

data = pd.read_csv(file, sep=';')


# Calculate correlation matrix
correlation_matrix = data.corr()

# Filter top correlations with 'Sana'
correlation_health = correlation_matrix['Sana'].sort_values(ascending=False)
correlation_health_top = correlation_health[abs(correlation_health) > 0.2]

# Select only the wavelengths
correlated_wavelengths = correlation_health_top.index[1:]
correlation_subset = correlation_matrix.loc[correlated_wavelengths, ['Sana']]

# Plot heatmap
plt.figure(figsize=(10, 12))
sns.heatmap(correlation_subset, annot=True, cmap="coolwarm", center=0, cbar_kws={'label': 'Correlación'}, fmt=".2f")
plt.title("Correlación entre 'Sana' y Longitudes de Onda Seleccionadas")
plt.show()

