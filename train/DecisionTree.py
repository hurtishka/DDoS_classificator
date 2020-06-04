import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pickle import dump
import numpy as np
from sklearn.model_selection import GridSearchCV

nd = pd.read_csv('new_data.csv')
nd.drop(['Unnamed: 0'], axis='columns', inplace=True)

Y = nd['Label'].astype('int')
X = nd.drop(['Label'], axis=1)


X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.3, random_state=1)

tree_params = {'max_depth': np.arange(1, 11), 'max_features':[.5, .7, 1]}

tree = DecisionTreeClassifier()
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1)

tree_grid.fit(X_train, Y_train)

prediction_tree = tree_grid.predict(X_valid)

print(accuracy_score(Y_valid, prediction_tree))

DT_model = input("Filename for saving? >")
dump(tree_grid, open(DT_model, "wb"))