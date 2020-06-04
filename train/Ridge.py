import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from pickle import dump
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)
ridge_params = {'alpha': np.arange(0.01,0.2)}

ridge = Ridge()
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, n_jobs=-1)
ridge_grid.fit(X_train, Y_train)

pred_ridge = ridge_grid.predict(X_valid)
print(r2_score(Y_valid, pred_ridge))

ridge_model = input("Filename for saving? >")
dump(ridge_grid, open(ridge_model, "wb"))
