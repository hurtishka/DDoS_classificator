import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from pickle import dump
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)

km_params = {'n_clusters': np.arange(1,8)}

km = KMeans()
km_grid = GridSearchCV(km, km_params, cv=5, n_jobs=-1)
km_grid.fit(X_train, Y_train)

pred_km = km_grid.predict(X_valid)
print(accuracy_score(Y_valid, pred_km))

km_model = input("Filename for saving? >")
dump(km_grid, open(km_model, "wb"))
