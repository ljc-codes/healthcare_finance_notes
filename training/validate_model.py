import os
from datetime import datetime

from pymongo import MongoClient
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from utils import constants
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
                           threshold=0.5,
                           feature_engineering_config_path='random_forest/feature_engineering_config.json',
                           vectorizer_folder='random_forest/vectorizers',
                           pca_folder='random_forest/pca',
                           model_folder='random_forest/models'):
    """
    Validates the performance of a random forest model

    Arguments:
        data_path (str): filepath to jsonl file with test dataset
        vectorizer_name (str): name of vectorizer file
        pca_name (str): name of pca file
        model_name (str): name of model file
    """

    # load and format test data
    X_test, y_test, feature_engineering_config = get_feature_set(
        data_path=data_path,
        vectorizer_name=vectorizer_name,
        pca_name=pca_name,
        feature_engineering_config_path=feature_engineering_config_path,
        vectorizer_folder=vectorizer_folder,
        pca_folder=pca_folder)

    model = file_system.load_component(function=None,
                                       data_input=None,
                                       component_folder=model_folder,
                                       component_name=model_name,
                                       component_config=None,
                                       label='model')
    y_pred = model.predict_proba(X_test)
    y_pred_val = model.predict(X_test)

    scores = {
        "auc": '{:.4f}'.format(roc_auc_score(y_true=y_test, y_score=y_pred[:, 1])),
        "accuracy": '{:.4f}'.format(accuracy_score(y_true=y_test, y_pred=y_pred_val)),
        "precision": '{:.4f}'.format(precision_score(y_true=y_test, y_pred=y_pred_val)),
        "recall": '{:.4f}'.format(recall_score(y_true=y_test, y_pred=y_pred_val))
    }

    model_result = create_model_result_document(model_type='random_forest',
                                                scores=scores,
                                                metadata=feature_engineering_config,
                                                data_file=data_path)

    store_model_result(model_result=model_result)


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

    args = parser.parse_args()

    validate_random_forest(data_path=args.data_path,
                           vectorizer_name=args.vectorizer_name,
                           pca_name=args.pca_name,
                           model_name=args.model_name)
