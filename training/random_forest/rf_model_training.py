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
                            cv=cv_folds,
                            verbose=5)
    model_cv.fit(X_train, y_train)

    # select best performing model
    best_model = model_cv.best_estimator_

    return best_model


def fit_random_forest(data_input,
                      config):
    """
    Fits a random forest model to training data

    Arguments:
        data_input (numpy array): training dataset
        config (dict): config params for a given model

    Returns:
        model (RandomForest Model)
    """
    model = RandomForestClassifier(**config)
    model.fit(*data_input)
    return model


def train_random_forest(data_path,
                        vectorizer_name,
                        pca_name,
                        model_name,
                        feature_engineering_config_path,
                        grid_search_config_path,
                        model_parameters_config_path,
                        vectorizer_folder,
                        pca_folder,
                        model_folder,
                        grid_search=False):
    """
    Loads data, transforms it with tf-idf (training a new vectorizer if necessary), grid searches
    over random forest hyperparameters for the best model, and saves the model as pickle object

    Arguments:
        data_path (str): filepath to jsonl file with training dataset
        vectorizer_name (str): name of vectorizer file
        pca_name (str): name of pca file
        model_name (str): name of model file
        feature_engineering_config_path (str): path to feature engineering (tfidf, pca) config file
        grid_search_config_path (str): path to grid search config file
        model_parameters_config_path (str): path to model parameters config
        vectorizer_folder (str): folder with vectorizer in it
        pca_folder (str): folder with pca in it
        model_folder (str): folder with model in it

    Keyword Arguments:
        grid_search (bool): run grid search over the model
    """
    # get training set
    X_train, y_train, _ = get_feature_set(
        data_path=data_path,
        feature_engineering_config_path=feature_engineering_config_path,
        vectorizer_folder=vectorizer_folder,
        vectorizer_name=vectorizer_name,
        pca_folder=pca_folder,
        pca_name=pca_name)

    if grid_search:

        # load grid search parameters
        with open(grid_search_config_path, "r") as f:
            model_parameters = json.load(f)

        print("Grid searching model...")
        model = grid_search_model(X_train=X_train,
                                  y_train=y_train,
                                  model_parameters=model_parameters)

    else:

        # load model parameters
        with open(model_parameters_config_path, "r") as f:
            model_parameters = json.load(f)

        model = file_system.load_component(function=fit_random_forest,
                                           data_input=(X_train, y_train),
                                           component_folder=model_folder,
                                           component_name=model_name,
                                           component_config=model_parameters,
                                           label='model')
        model.fit(X_train, y_train)

    model_save_path = os.path.join(model_folder, model_name) + '.pkl'
    file_system.save_component(save_folder=model_folder,
                               save_path=model_save_path,
                               component=model)

    print("Complete!")


def main():
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

    parser.add_argument('--pca_name',
                        '-p',
                        required=True,
                        type=str,
                        help='Name of pca, if none exists will be created')

    parser.add_argument('--model_name',
                        '-mn',
                        required=True,
                        type=str,
                        help='Name of model')

    parser.add_argument('--grid_search',
                        '-g',
                        action='store_true',
                        default=False,
                        help='Grid search model parameters')

    parser.add_argument('--feature_engineering_config_path',
                        '-fc',
                        default=os.path.join(os.getcwd(), 'random_forest/feature_engineering_config.json'),
                        type=str,
                        help='Path to feature engineering configuration file')

    parser.add_argument('--grid_search_config_path',
                        '-gc',
                        default=os.path.join(os.getcwd(), 'random_forest/rf_grid_search_config.json'),
                        type=str,
                        help='Path to grid search configuration file')

    parser.add_argument('--model_parameters_config_path',
                        '-mc',
                        default=os.path.join(os.getcwd(), 'random_forest/rf_model_config.json'),
                        type=str,
                        help='Path to model parameters configuration file')

    parser.add_argument('--vectorizer_folder',
                        '-vf',
                        default=os.path.join(os.getcwd(), 'random_forest/vectorizers'),
                        type=str,
                        help='Path to vectorizer folder')

    parser.add_argument('--pca_folder',
                        '-pf',
                        default=os.path.join(os.getcwd(), 'random_forest/pca'),
                        type=str,
                        help='Path to pca folder')

    parser.add_argument('--model_folder',
                        '-mf',
                        default=os.path.join(os.getcwd(), 'random_forest/models'),
                        type=str,
                        help='Path to model folder')

    args = parser.parse_args()

    train_random_forest(data_path=args.data_path,
                        vectorizer_name=args.vectorizer_name,
                        pca_name=args.pca_name,
                        model_name=args.model_name,
                        feature_engineering_config_path=args.feature_engineering_config_path,
                        grid_search_config_path=args.grid_search_config_path,
                        model_parameters_config_path=args.model_parameters_config_path,
                        vectorizer_folder=args.vectorizer_folder,
                        pca_folder=args.pca_folder,
                        model_folder=args.model_folder,
                        grid_search=args.grid_search)
