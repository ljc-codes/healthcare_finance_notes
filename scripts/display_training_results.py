import os
import json

from pymongo import MongoClient

from notetagger import constants


def display_training_results():
    """
    displays model results in mongo db database
    """

    # get collection
    client = MongoClient(os.environ["MONGO_CONFIG"])
    collection = client[constants.MONGO_DATABASE_NAME][constants.MONGO_COLLECTION_NAME]

    # print out results
    for doc in collection.find():
        if 'performance_metrics' in doc:
            print('model id: ', doc['model_id'])
            print(json.dumps(doc['performance_metrics'], indent=4))
            print('\n')
