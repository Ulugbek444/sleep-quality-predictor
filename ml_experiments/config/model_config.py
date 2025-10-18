KNN_PARAMS = {
    'model__n_neighbors': [3, 5, 7, 9, 11],              # количество соседей
    'model__weights': ['uniform', 'distance'],           # веса
    'model__metric': ['euclidean', 'manhattan'],         # метрика
    'model__p': [1, 2],                                  # степень метрики Minkowski
    'model__algorithm': ['auto', 'kd_tree', 'ball_tree']  # способ поиска соседей
}


NAIVE_BAYES_PARAMS = {
    'model__var_smoothing': [1e-12, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
}

LOGISTIC_REGRESSION_PARAMS = {
    "model__C": [0.01, 0.1, 1, 10],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
    "model__max_iter": [1000]
}

RANDOM_FOREST_PARAMS = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5]
}

XGBOOST_PARAMS = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [3, 6],
    "model__learning_rate": [0.01, 0.1],
}
