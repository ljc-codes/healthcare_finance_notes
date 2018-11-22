import os

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
        for key in doc:
            print(key, doc[key])
        print('\n')


if __name__ == "__main__":
    display_training_results()
