import os

from pymongo import MongoClient

from utils import constants


def display_results(score="auc"):

    # get collection
    client = MongoClient(os.environ["MONGO_CONFIG"])
    collection = client[constants.MONGO_DATABASE_NAME][constants.MONGO_COLLECTION_NAME]

    # print out results
    for doc in collection.find():
        for key in doc:
            if key not in ['metadata', '_id', 'date']:
                print(key, doc[key])
        print('\n')


if __name__ == "__main__":
    display_results()
