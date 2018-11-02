import os
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from training.random_forest.training_preparation import get_feature_set
from training import file_system


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


def train_random_forest(data_path,
                        vectorizer_name,
                        pca_name,
                        model_name,
                        feature_engineering_config_path='feature_engineering_config.json',
                        grid_search_config_path='rf_grid_search_config.json',
                        vectorizer_folder='vectorizers',
                        pca_folder='pca',
                        models_folder='models'):
    """
    Loads data, transforms it with tf-idf (training a new vectorizer if necessary), grid searches
    over random forest hyperparameters for the best model, and saves the model as pickle object
    """
    # get training set
    X_train, y_train, feature_engineering_config = get_feature_set(
        data_path=data_path,
        feature_engineering_config_path=feature_engineering_config_path,
        vectorizer_folder=vectorizer_folder,
        vectorizer_name=vectorizer_name,
        pca_folder=pca_folder,
        pca_name=pca_name)

    # load grid search parameters
    with open(grid_search_config_path, "r") as f:
        model_parameters = json.load(f)

    print("Grid searching model...")
    model = grid_search_model(X_train=X_train,
                              y_train=y_train,
                              model_parameters=model_parameters)

    model_save_path = os.path.join(models_folder, model_name) + '.pkl'

    file_system.save_component(save_folder=models_folder,
                               save_path=model_save_path,
                               component=model)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        '-d',
                        required=True,
                        type=str,
                        help='Path to data file')

    parser.add_argument('--vectorizer_name',
                        '-v',
                        required=True,
                        type=str,
                        help='Name of vectorizer, if none exists will be created')

    parser.add_argument('--pca',
                        '-p',
                        required=True,
                        type=str,
                        help='Name of pca, if none exists will be created')

    parser.add_argument('--model_name',
                        '-mn',
                        required=True,
                        type=str,
                        help='Name of model')

    args = parser.parse_args()

    train_random_forest(data_path=args.data_path,
                        vectorizer_name=args.vectorizer_name,
                        pca_name=args.pca_name,
                        model_name=args.model_name)
