"""
Tuning_hyper-parameters.py   

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm

import pickle

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
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

parameters = {'kernel':('linear', 'rbf', 'poly', 'sigmoid'), 'C':[0.1, 1, 10, 50]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(X_train, y_train)


file2 = 'SVC_GridSearch.pkl'
file2 = os.path.join(path, file2)
with open(file2, 'wb') as f:
    pickle.dump(clf, f)


print(clf.score(X_test, y_test))
print()
print ("best parameters: {}".format (clf.best_params_))




