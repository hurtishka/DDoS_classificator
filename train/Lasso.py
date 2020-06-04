import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from pickle import dump
from sklearn.metrics import r2_score
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)
lasso_params = {'alpha': np.arange(0.01,0.2)}

lasso = Lasso()
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, n_jobs=-1)
lasso_grid.fit(X_train, Y_train)

pred_lasso = lasso_grid.predict(X_valid)
print(r2_score(Y_valid, pred_lasso))

lasso_model = input("Filename for saving? >")
dump(lasso_grid, open(lasso_model, "wb"))