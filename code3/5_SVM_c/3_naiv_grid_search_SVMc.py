"""
3_naiv_grid_search_SVMc.py

"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = "datos.csv"
full_file = os.path.join(path, file)

df = pd.read_csv(full_file, sep=';')

X_df = df.iloc[:, 4:]
y_df = df['Sana']

df = None

X_train, X_test, y_train, y_test = train_test_split(
    X_df, 
    y_df, 
    test_size=0.33, 
    random_state=96
)

X_df, y_df = None, None

print('X_train size:', X_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', X_test.shape)
print('y_test size:', y_test.shape)
print()


best_score = 0
best_clf = None
for C in [0.001, 0.01, 0.1, 1, 10, 100]:
    for gamma in ['scale', 0.5, 1, 10]:
        for r in [0, 1, 5, 20]:
            clf = SVC(kernel='rbf', C=C, gamma=gamma, coef0=r)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            print('score:', score)
            print('C:', C)
            print('gamma:', gamma)
            print('coef0', r)
            print()

            if score > best_score:
                best_score = score
                best_parameters = {
                    'kernel':'rbf',
                    'C':C,
                    'gamma':gamma,
                    'coef0':r
                }
print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameters))