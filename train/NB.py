import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from pickle import dump
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)

nb_params = {'var_smoothing': np.arange(1e-9, 1e-8)}

nb = GaussianNB()
nb_grid = GridSearchCV(nb, nb_params, cv=5, n_jobs=-1)
nb_grid.fit(X_train, Y_train)

pred_nb = nb_grid.predict(X_valid)
print(accuracy_score(Y_valid, pred_nb))

nb_model = input("Filename for saving? >")
dump(nb_grid, open(nb_model, "wb"))