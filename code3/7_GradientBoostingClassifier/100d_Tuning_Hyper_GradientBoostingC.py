"""
100d_Tuning_Hyper_GradientBoostingC.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

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
    random_state=57
)

best_score = 0

for loss in ['log_loss', 'exponential']:
    for n_estimators in [800, 1000]:
        for criterion in ['friedman_mse', 'squared_error']:
            for max_depth in [3, 7, 15, 24]:
                for max_features in ['sqrt', 'log2']:
                    clf = GradientBoostingClassifier(
                        loss=loss,
                        n_estimators=n_estimators,
                        criterion=criterion,
                        max_depth=max_depth,
                        max_features=max_features
                    )
                    clf.fit(X_train, y_train)
                    score = clf.score(X_test, y_test)

                    print('score:', score)
                    print('loss:', loss)
                    print('n_estimators', n_estimators)
                    print('criterion:', criterion)
                    print('max_depth:', max_depth)
                    print('max_features:', max_features)
                    print()

                    if score > best_score:
                        best_score = score
                        best_parameters = {
                            'score:':score,
                            'loss:':loss,
                            'n_estimators':n_estimators,
                            'criterion:':criterion,
                            'max_depth:':max_depth,
                            'max_features:': max_features
                        }




print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameters))


"""

print()
print("Best score: {}".format(best_score))
print("Best parameters: {}".format(best_parameters))

"""
                

