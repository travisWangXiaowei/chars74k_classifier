
import numpy as np
import os
import cv2
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
parameter_space = {
    'hidden_layer_sizes': [(129,400,62)],
    'activation': ['relu'],
    'solver': ['adam'],
    'alpha': [0.0001, 0.05,0.005,0.0005],
    'max_iter':[200],
    'verbose' :[10]
}
mlp = MLPClassifier(max_iter=200)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X,y)
# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# 0.557 (+/-0.010) for {'hidden_layer_sizes': (129, 400, 62), 'solver': 'adam', 'max_iter': 200, 'activation': 'relu', 'alpha': 0.0001, 'verbose': 10}
# 0.561 (+/-0.014) for {'hidden_layer_sizes': (129, 400, 62), 'solver': 'adam', 'max_iter': 200, 'activation': 'relu', 'alpha': 0.05, 'verbose': 10}
# 0.555 (+/-0.014) for {'hidden_layer_sizes': (129, 400, 62), 'solver': 'adam', 'max_iter': 200, 'activation': 'relu', 'alpha': 0.005, 'verbose': 10}
# 0.558 (+/-0.028) for {'hidden_layer_sizes': (129, 400, 62), 'solver': 'adam', 'max_iter': 200, 'activation': 'relu', 'alpha': 0.0005, 'verbose': 10}
