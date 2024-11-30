"""
SVMc_linear.py

"""


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
X_df = df.iloc[:, 4:].values
y_df = df['Sana'].values
df = None

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.3,
    random_state=96
)

X_df, y_df = None, None

C = 0.3020408163265306
clf = SVC(C=C, kernel='linear')

clf.fit(X_train, y_train)

score_train = clf.score(X_train, y_train)
score_test = clf.score(X_test, y_test)

print('score en train:', score_train)
print('score en test:', score_test)

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code4\Sets"
path = path.replace('\\', '/')

file = 'model_SVMc_linear.pkl'
file = os.path.join(path, file)

with open(file, 'wb') as f:
    pickle.dump(clf, f)
