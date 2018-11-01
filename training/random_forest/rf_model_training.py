import os
import json
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from training.training_preparation import get_tf_idf_set


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
                        vectorizer_folder,
                        vectorizer_name,
                        tfidf_config_path,
                        grid_search_config_path,
                        models_folder,
                        model_name):
    """
    Loads data, transforms it with tf-idf (training a new vectorizer if necessary), grid searches
    over random forest hyperparameters for the best model, and saves the model as pickle object
    """
    # get training set
    X_train, y_train = get_tf_idf_set(data_path=data_path,
                                      vectorizer_folder=vectorizer_folder,
                                      vectorizer_name=vectorizer_name,
                                      tfidf_config_path=tfidf_config_path)

    # load grid search parameters
    with open(grid_search_config_path, "r") as f:
        model_parameters = json.load(f)

    print("Grid searching model...")
    model = grid_search_model(X_train=X_train,
                              y_train=y_train,
                              model_parameters=model_parameters)

    # create model folder if it does not exist
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    model_save_path = os.path.join(models_folder, model_name) + '.pkl'
    # save model
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path',
                        '-d',
                        required=True,
                        type=str,
                        help='Path to data file')

    parser.add_argument('--vectorizer_folder',
                        '-vf',
                        default='../vectorizers',
                        type=str,
                        help='Path to vectorizer folder, if none exists will be created')

    parser.add_argument('--vectorizer_name',
                        '-vn',
                        required=True,
                        type=str,
                        help='Name of vectorizer, if none exists will be created')

    parser.add_argument('--tfidf_config_path',
                        '-t',
                        default='../tf_idf_config.json',
                        type=str,
                        help='Path to tf-idf config file')

    parser.add_argument('--grid_search_config_path',
                        '-g',
                        default='rf_grid_search_config.json',
                        type=str,
                        help='Path to grid search config file')

    parser.add_argument('--models_folder',
                        '-mf',
                        default='models',
                        type=str,
                        help='Path to models, if none exists it will be created')

    parser.add_argument('--model_name',
                        '-mn',
                        required=True,
                        type=str,
                        help='Name of model')

    args = parser.parse_args()

    train_random_forest(data_path=args.data_path,
                        vectorizer_folder=args.vectorizer_folder,
                        vectorizer_name=args.vectorizer_name,
                        tfidf_config_path=args.tfidf_config_path,
                        grid_search_config_path=args.grid_search_config_path,
                        models_folder=args.models_folder,
                        model_name=args.model_name)
