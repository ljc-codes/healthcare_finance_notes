import os
from datetime import datetime

from pymongo import MongoClient

from utils import constants


def stores_model_result(model_result,
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
                                 score_type,
                                 score,
                                 metadata):
    """
    Creates dictionary of model results

    Arguments:
        model_type (str): type of model (e.g. `random_forest`)
        score_type (str): type of score (e.g. `auc`)
        score (float): value of score
        metadata (dict): additional data to be included with model result

    Returns:
        model_summary (dict): dictionary holding various info on the model result
    """
    model_summary = {
        "model_type": model_type,
        score_type: '{0:.4f}'.format(score),
        "metadata": metadata,
        "date": datetime.utcnow()
    }

    return model_summary
