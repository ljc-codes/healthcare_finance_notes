import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def grid_search_model(X_train,
                      y_train,
                      model_parameters,
                      scoring_metric="roc_auc",
                      cv_folds=5):
    """
    Trains a random forest, grid searching over provided parameters and returning the best one

    Arguments:
        X_train (numpy array): training features
        y_train (numpy array): training outcomes
        model_parameters (dict): parameters to grid search over

    Keyword Arguments:
        scoring_metric (str): metric to optimize grid search over
        cv_folds (int): number of folds to use in calculating metric

    Returns:
        best_model (Sklearn Model): sklearn model object to use for predictions
    """

    # initialize model
    model = RandomForestClassifier()

    # initialize grid search object
    model_cv = GridSearchCV(estimator=model,
                            param_grid=model_parameters,
                            scoring=scoring_metric,
                            cv=cv_folds)
    model_cv.fit(X_train, y_train)

    # select best performing model
    best_model = model_cv.best_estimator_

    return best_model


def train_random_forest(window_size,
                        grid_search_config_path,
                        model_save_path):

    # load grid search parameters
    with open(grid_search_config_path, "r") as f:
        model_parameters = json.load(f)

    model = grid_search_model(X_train=X_train,
                              y_train=y_train,
                              model_parameters=model_parameters)


