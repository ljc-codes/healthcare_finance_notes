import os
import json
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from predict import convert_to_tfidf


def fit_tf_idf(train_df,
               feature_column_name='tokenized_snippet',
               vectorizer_args={}):
    """
    Fits a tf-idf vectorizer on a provided dataframe

    Arguments:
        train_df (Pandas Dataframe): dataframe with feature labels

    Keyword Arguments:
        feature_column_name (str): column name of feature to convert into a tf-idf matrix

    Returns:
        vectorizer (TfidfVectorizer): vectorizer for transforming any dataframe
    """
    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=lambda doc: doc,
                                 preprocessor=lambda doc: doc,
                                 token_pattern=None,
                                 **vectorizer_args)

    vectorizer.fit(train_df[feature_column_name])

    return vectorizer


def get_tf_idf_set(data_path,
                   vectorizer_path,
                   tfidf_config_path,
                   feature_column_name="tokenized_snippet",
                   outcome_column_name="y"):
    """
    Loads dataset and converts it into a tfidf matrix and outcome array

    Arguments:
        data_path (str): filepath to jsonl file with dataset
        vectorizer_path (str): filepath to tfidf vectorizer, if the file does not exist a new one is created
        tfidf_path (str): filepath to tfidf config, only used if vectorizer created

    Returns:
        X (numpy array): tfidf matrix
        y (numpy array): outcome labels
    """

    # load data
    df = pd.read_json(data_path,
                      orient='records',
                      lines=True)

    # check if vectorizer file exists, if so load it, otherwise create it
    if os.path.isfile(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        # load tfidf parameters
        with open(tfidf_config_path, "r") as f:
            tfidf_parameters = json.load(f)
            if "ngram_range" in tfidf_parameters:
                tfidf_parameters["ngram_range"] = tuple(tfidf_parameters["ngram_range"])

        # fit vectorizer
        vectorizer = fit_tf_idf(train_df=df,
                                feature_column_name=feature_column_name,
                                vectorizer_args=tfidf_parameters)

        # save vectorizer
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)

    X = convert_to_tfidf(df=df,
                         vectorizer=vectorizer,
                         feature_column_name=feature_column_name)
    y = df[outcome_column_name].values

    return X, y
