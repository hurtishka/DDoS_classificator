import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pickle import dump
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)
svm_params = {'degree': np.arange(5,9)}

svm = SVC(kernel='poly', gamma='auto')

svm_grid = GridSearchCV(svm, svm_params, cv=5, n_jobs=-1)
svm_grid.fit(X_train, Y_train)

pred_svm = svm_grid.predict(X_valid)
print(accuracy_score(Y_valid, pred_svm))

svm_model = input("Filename for saving? >")
dump(svm_grid, open(svm_model, "wb"))