"""


"""


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from plot_model import plot_clf_model, plot_reg_model



def get_datos(random_state, etiqueta):
    path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
    path = path.replace("\\", "/")

    file = 'datos.csv'
    file = os.path.join(path, file)

    df = pd.read_csv(file, sep=';')
    X_df = df.iloc[:, 4:].values
    y_df = df[etiqueta].values

    X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.3,
    random_state=random_state
    )   

    return X_test, y_test

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code4\Sets"
path = path.replace("\\", "/")




file = 'model_Ridge_linear_reg.pkl'
file = os.path.join(path, file)

with open(file, 'rb') as f:
    reg = pickle.load(f)

X_data, y_data = get_datos(random_state=203, etiqueta='dpi')

plot_reg_model(reg, X_data, y_data)



"""
file = 'model_SVMc_linear.pkl'
file = os.path.join(path, file)


with open(file, 'rb') as f:
    clf = pickle.load(f)

X_data, y_data = get_datos(random_state=96, etiqueta='Sana')


plot_clf_model(clf, X_data, y_data)


"""


