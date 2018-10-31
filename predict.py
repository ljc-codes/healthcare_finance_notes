

def convert_to_tfidf(df,
                     vectorizer,
                     feature_column_name='tokenized_snippet'):
    """
    Converts a chosen column in a dataframe to a tf-idf matrix

    Arguments:
        df (Pandas Dataframe): dataframe with feature labels
        vectorizer (TfidfVectorizer): vectorizer for transforming any dataframe

    Keyword Arguments:
        feature_column_name (str): column name of feature to convert into a tf-idf matrix

    Returns:
        X (numpy array): numpy array with dimensions docs x terms
    """
    X = vectorizer.transform(df[feature_column_name])
    return X
