"""
100d_RandomForestClassifier_tuning_hyperparametrs.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace("\\", "/")

file1 = 'datos_PCA_100d.csv'
file1 = os.path.join(path, file1)

df = pd.read_csv(file1, sep=';')

X_df = df.iloc[:, 4:]
y_df = df['Sana']

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.33,
    random_state=47
)

best_score = 0

for n_estimators in [600, 800, 1000]:
    for max_depth in [7, 14, 21, 30]:
        for criterion in ['entropy', 'gini', 'log_loss']:
            for max_features in ['sqrt', 'log2']:
                clf = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    criterion=criterion, 
                    max_features=max_features
                )
                clf.fit(X_train, y_train)
                score = clf.score(X_test, y_test)

                print('score:', score)
                print('n_estimators:', n_estimators)
                print('max_depth:', max_depth)
                print('criterion:', criterion)
                print('max_features:', max_features)
                print()

                if score > best_score:
                    best_score = score
                    best_parameters = {
                        'n_estimators':n_estimators,
                        'max_depth':max_depth,
                        'criterion':criterion,
                        'max_features':max_features
                    }

                
print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameters))





"""
clf = RandomForestClassifier(
    n_estimators=1000,
    criterion="entropy",
    max_depth=7,
    max_features="sqrt",
)


"""







