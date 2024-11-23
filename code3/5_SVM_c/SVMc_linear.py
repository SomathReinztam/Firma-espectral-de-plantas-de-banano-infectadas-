"""
SVMc_linear.py

"""

import os
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = 'datos.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')
X_df = df.iloc[:, 4:]
y_df = df['Sana']
df = None


def get_datos(semilla):
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.33, random_state=semilla)
    return X_train, X_test, y_train, y_test

semillas = [2, 7, 15, 21, 22, 38, 53, 58, 72, 75, 88, 10, 93, 100, 33, 103, 44, 107, 117, 122, 130, 131, 150, 152, 158, 161, 180, 188, 189, 192]

C = 0.3020408163265306
for i in semillas:
    clf = SVC(C=C, kernel='linear')
    X_train, X_test, y_train, y_test = get_datos(semilla=i)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(score, i)


X_train, X_test, y_train, y_test = get_datos(semilla=96)
clf.fit(X_train, y_train)

file = 'SVMc_linear_model.pkl'
file = os.path.join(path, file)

with open(file, 'wb') as f:
    pickle.dump(clf, f)

