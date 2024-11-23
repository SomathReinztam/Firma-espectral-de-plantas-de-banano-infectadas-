"""
grid_search_LogisticRegression.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file = 'datos.csv'
full_file = os.path.join(path, file)

df = pd.read_csv(full_file, sep=';')

X_df = df.iloc[:, 4:]

y_df = df['Sana']

X_train, X_test, y_train, y_test = train_test_split(
    X_df, 
    y_df, 
    test_size=0.33, 
    random_state=10
)

best_score = 0

for penalty in ['l2', None]:
    for C in [0.001, 0.01, 0.1, 1, 10, 100]:
        clf = LogisticRegression(
            penalty=penalty, 
            C=C,
            max_iter=2000
        )

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        print('score:', score)
        print('penalty:', penalty)
        print()
        if score > best_score:
            best_score = score
            best_parameters = {
                'penalty':penalty,
                'C':C,
                'max_iter':2000
            }


print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameters))

