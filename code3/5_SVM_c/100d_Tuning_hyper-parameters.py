"""
100d_Tuning_hyper-parameters.py

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos_PCA_100d.csv"
full_file = os.path.join(path, file)

df = pd.read_csv(full_file, sep=';')

X_df = df.iloc[:, 4:]
y_df = df['Sana']

X_train, X_test, y_train, y_test = train_test_split(
    X_df, 
    y_df, 
    test_size=0.33, 
    random_state=26
)

