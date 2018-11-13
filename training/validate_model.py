import os
from datetime import datetime

from pymongo import MongoClient
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from notetagger import constants
from training import file_system
from training.random_forest.training_preparation import get_feature_set


def store_model_result(model_result,
                       db_name=constants.MONGO_DATABASE_NAME,
                       collection_name=constants.MONGO_COLLECTION_NAME):
    """
    Store model in MongoDB and prints id

    Arguments:
        model_result (dict): dictionary holding various info on the model result

    Keyword Arguments:
        db_name (str): name of database in MongoDB
        collection_name (str): name of collection in database
    """
    client = MongoClient(os.environ["MONGO_CONFIG"])
    db = client[db_name]
    collection = db[collection_name]
    result_id = collection.insert_one(model_result).inserted_id
    print("Result {} saved".format(result_id))


def create_model_result_document(model_type,
                                 scores,
                                 metadata,
                                 data_file):
    """
    Creates dictionary of model results

    Arguments:
        model_type (str): type of model (e.g. `random_forest`)
        scores (dict): dict of score_type: score_value
        metadata (dict): additional data to be included with model result
        data_file (str): path to data file validated on

    Returns:
        model_result (dict): dictionary holding various info on the model result
    """
    model_result = {
        "model_type": model_type,
        "metadata": metadata,
        "date": datetime.utcnow(),
        "data": data_file
    }

    # add scores
    for score in scores:
        model_result[score] = scores[score]

    return model_result


def validate_random_forest(data_path,
                           vectorizer_name,
                           pca_name,
                           model_name,
                           feature_engineering_config_path,
                           vectorizer_folder,
                           pca_folder,
                           model_folder,
                           window_size,
                           threshold=0.5):
    """
    Calculates various metrics of performance on the random forest model and saves to MongoDB

    Arguments:
        data_path (str): filepath to jsonl file with test dataset
        vectorizer_name (str): name of vectorizer file
        pca_name (str): name of pca file
        model_name (str): name of model file
        feature_engineering_config_path (str): path to feature engineering (tfidf, pca) config file
        vectorizer_folder (str): folder with vectorizer in it
        pca_folder (str): folder with pca in it
        model_folder (str): folder with model in it
        window_size (int): size of window around word tag to extract from data

    Keyword Arguments:
        threshold (float): threshold for positive class
    """

    # load and format test data
    X_test, y_test, feature_engineering_config = get_feature_set(
        data_path=data_path,
        vectorizer_name=vectorizer_name,
        pca_name=pca_name,
        feature_engineering_config_path=feature_engineering_config_path,
        vectorizer_folder=vectorizer_folder,
        pca_folder=pca_folder,
        window_size=window_size)

    model = file_system.load_component(function=None,
                                       data_input=None,
                                       component_folder=model_folder,
                                       component_name=model_name,
                                       component_config=None,
                                       label='model')

    print("Making predictions...")
    y_pred = model.predict_proba(X_test)
    y_pred_val = model.predict(X_test)

    scores = {
        "auc": '{:.4f}'.format(roc_auc_score(y_true=y_test, y_score=y_pred[:, 1])),
        "accuracy": '{:.4f}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_val)),
        "precision": '{:.4f}'.format(precision_score(y_true=y_test, y_pred=y_pred_val)),
        "recall": '{:.4f}'.format(recall_score(y_true=y_test, y_pred=y_pred_val))
    }

    print("Storing results...")
    model_result = create_model_result_document(model_type='random_forest',
                                                scores=scores,
                                                metadata=feature_engineering_config,
                                                data_file=data_path)

    store_model_result(model_result=model_result)


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

    parser.add_argument('--feature_engineering_config_path',
                        '-f',
                        default=os.path.join(os.getcwd(), 'random_forest/feature_engineering_config.json'),
                        type=str,
                        help='Path to feature engineering configuration file')

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

    parser.add_argument('--window_size',
                        '-w',
                        required=True,
                        type=int,
                        help='Size of window around tags')

    args = parser.parse_args()

    validate_random_forest(data_path=args.data_path,
                           vectorizer_name=args.vectorizer_name,
                           pca_name=args.pca_name,
                           model_name=args.model_name,
                           feature_engineering_config_path=args.feature_engineering_config_path,
                           vectorizer_folder=args.vectorizer_folder,
                           pca_folder=args.pca_folder,
                           model_folder=args.model_folder,
                           window_size=args.window_size)
