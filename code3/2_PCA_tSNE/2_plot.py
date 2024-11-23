"""
2_plot.py

"""
import os
import pandas as pd
import matplotlib.pyplot as plt

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos_describe.csv"
full_path = os.path.join(path, file)

df = pd.read_csv(full_path, sep=';')



def plot_std():
    lgtd_onda = [float(i) for i in df.columns[5:]]

    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(10,6))
    ax = plt.axes()

    y = df.iloc[2, 5:].values
    ax.plot(lgtd_onda, y, label="varianza")

    plt.show()

plot_std()