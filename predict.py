from utils import constants


def predict_tfidf_model(df,
                        vectorizer,
                        model,
                        feature_column_name=constants.FEATURE_COLUMN_NAME,
                        predictions_column_name=constants.PREDICTION_COLUMN_NAME):
    """
    Converts a chosen column in a dataframe to a tf-idf matrix and then predicts
    with a given model, storing the predictions in another column

    Arguments:
        df (Pandas Dataframe): dataframe with feature labels
        vectorizer (TfidfVectorizer): vectorizer for transforming any dataframe
        model (Sklearn model): object that has a `predict` function

    Keyword Arguments:
        feature_column_name (str): column name of feature to convert into a tf-idf matrix
        predictions_column_name (str): column name to save predictions in

    Returns:
        df (Pandas Dataframe): dataframe with feature labels and predictions
    """
    X = vectorizer.transform(df[feature_column_name])
    df[predictions_column_name] = model.predict(X)
    return df
