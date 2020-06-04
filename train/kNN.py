import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from pickle import dump
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)

knn_params = {'n_neighbors': np.arange(88, 90)}

knn = KNeighborsClassifier()

knn_grid = GridSearchCV(knn, knn_params, cv=3, n_jobs=-1)
knn_grid.fit(X_train, Y_train)

pred_knn = knn_grid.predict(X_valid)
print(accuracy_score(Y_valid, pred_knn))

kNN_model = input("Filename for saving? >")
dump(knn_grid, open(kNN_model, "wb"))