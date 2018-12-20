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
    for doc in collection.find({'performance_metrics': {'$exists': True}, 'performance_metrics.auc': {'$gt': '0.95'}}):
        print('model id: ', doc['model_id'])
        doc['performance_metrics']['metrics_by_threshold'] = (
            [metric_group for metric_group in doc['performance_metrics']['metrics_by_threshold']
             if float(metric_group['precision']) > 0.9])
        print(json.dumps(doc['performance_metrics'], indent=4))
        print('\n')
