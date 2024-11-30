"""
Ridge_linear_reg.py

"""

import os 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle


path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\sets"
path = path.replace('\\', '/')

file = 'datos_no_control.csv'
file = os.path.join(path, file)

df = pd.read_csv(file, sep=';')

X_df = df.iloc[:, 4:].values
y_df = df['dpi'].values

df = None

X_train, X_test, y_train, y_test = train_test_split(
    X_df,
    y_df,
    test_size=0.3,
    random_state=93
)

X_df, y_df = None, None

reg = Ridge(alpha=7)
reg.fit(X_train, y_train)

path = r"C:\Users\Acer\Documents\python\Proyecto de investigacion\code4\Sets"
path = path.replace('\\', '/')

file = 'model_Ridge_linear_reg.pkl'
file = os.path.join(path, file)

with open(file, 'wb') as f:
    pickle.dump(reg, f)

y_train_predict = reg.predict(X_train)
y_test_predict = reg.predict(X_test)



print(
    'train mean_absolute_error:',
    mean_absolute_error(y_train, y_train_predict)
)
print()
print(
    'test mean_absolute_error:',
    mean_absolute_error(y_test, y_test_predict)
)