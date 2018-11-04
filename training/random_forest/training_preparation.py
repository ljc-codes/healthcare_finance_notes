import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from notetagger import constants
from training import file_system


def fit_tfidf(data_input,
              config,
              feature_column_name=constants.FEATURE_COLUMN_NAME):
    """
    Fits a tf-idf vectorizer on a provided dataframe

    Arguments:
        data_input (Pandas Dataframe): dataframe with feature labels
        config (dict): config params for a given vectorizer

    Keyword Arguments:
        feature_column_name (str): column name of feature to convert into a tf-idf matrix

    Returns:
        vectorizer (TfidfVectorizer): vectorizer for transforming any dataframe
    """

    vectorizer = TfidfVectorizer(analyzer='word',
                                 tokenizer=lambda doc: doc,
                                 preprocessor=lambda doc: doc,
                                 token_pattern=None,
                                 **config)

    vectorizer.fit(data_input[feature_column_name])

    return vectorizer


def fit_pca(data_input,
            config):
    """
    Fits an PCA component to a provided array

    Arguments:
        data_input (numpy array): matrix transformed by tfidf
        config (dict): config params for a given pca

    Returns:
        pca (TruncatedSVD): pca object for transforming any tfidf matrix
    """

    pca = TruncatedSVD(**config)

    pca.fit(data_input)

    return pca


def get_feature_set(data_path,
                    feature_engineering_config_path,
                    vectorizer_folder,
                    vectorizer_name,
                    pca_folder,
                    pca_name,
                    feature_column_name=constants.FEATURE_COLUMN_NAME,
                    outcome_column_name=constants.OUTCOME_COLUMN_NAME):
    """
    Loads dataset and converts it into a tfidf matrix and outcome array

    Arguments:
        data_path (str): filepath to jsonl file with dataset
        vectorizer_folder (str): folder to tfidf vectorizer, if the folder does not exist a new one is created
        vectorizer_name (str): name of vectorizer, if the vectorizer does not exist it is created
        tfidf_path (str): filepath to tfidf config, only used if vectorizer created

    Returns:
        X (numpy array): tfidf matrix
        y (numpy array): outcome labels
        feature_engineering_params (dict): dictionary of parameters used to transform data
    """

    # load data
    df = pd.read_json(data_path,
                      orient='records',
                      lines=True)

    # load tfidf and pca parameters
    with open(feature_engineering_config_path, "r") as f:
        feature_engineering_params = json.load(f)

        # format ngram range correctly
        if "ngram_range" in feature_engineering_params["tfidf"]:
            feature_engineering_params["tfidf"]["ngram_range"] = tuple(feature_engineering_params["tfidf"]
                                                                       ["ngram_range"])
    # format data with tfidf and pca

    vectorizer = file_system.load_component(
        function=fit_tfidf,
        data_input=df,
        component_folder=vectorizer_folder,
        component_name=vectorizer_name,
        component_config=feature_engineering_params["tfidf"],
        label="tfidf vectorizer")

    print("Transforming dataset with tfidf...")
    X_full = vectorizer.transform(df[feature_column_name])

    pca = file_system.load_component(
        function=fit_pca,
        data_input=X_full,
        component_folder=pca_folder,
        component_name=pca_name,
        component_config=feature_engineering_params["pca"],
        label="pca")

    print("Reducing dimensionality with pca...")
    X = pca.transform(X_full)

    y = df[outcome_column_name].values

    return X, y, feature_engineering_params
