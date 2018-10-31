import string

import nltk
from tqdm import tqdm
import pandas as pd

from utils import constants

STOPWORDS = set(nltk.corpus.stopwords.words('english'))
PUNCTUATION = set(string.punctuation)


def clean_text(text):
    """
    Removes punctuation, tokenizes text, lower cases it, and removes stop words

    Arguments:
        text (str): document

    Returns:
        clean_text (list of str): list of tokens
    """
    tokenized_text = nltk.word_tokenize(text)
    lower_case_words = [word.lower() for word in tokenized_text if word.isalpha()]
    clean_text = [word for word in lower_case_words if word not in STOPWORDS]
    return clean_text


def extract_text_snippets(tokenized_text, tag, window_size):
    """
    Gets list of tokens of window size * 2 surrounding a specific tag word

    Arguments:
        tokenized_text (list of str): list of tokens
        tag (str): word to get window around
        window_size (int): number of tokens to get both before and after the `tag`

    Returns:
        text_snippets (list of list of str): list of list of tokens of size window * 2
    """

    tag_indices = []
    for i, token in enumerate(tokenized_text):
        if token == tag:
            tag_indices.append(i)

    window_params = [(max(0, index - window_size), min(len(tokenized_text), index + window_size))
                     for index in tag_indices]

    text_snippets = [tokenized_text[param[0]:param[1]] for param in window_params]
    return text_snippets


def process_text(df,
                 window_size,
                 training=False,
                 text_column_name=constants.TEXT_COLUMN_NAME,
                 tags_column_name=constants.TAGS_COLUMN_NAME,
                 financial_flag_column_name=constants.FINANCIAL_FLAG_COLUMN_NAME):
    """
    Process text into trainable set of tokens and outcome labels

    Arguments:
        df (Pandas DataFrame): Pandas dataframe with note id, note text, and list of string tags
        window_size (int): number of tokens to get both before and after the `tag`

    Returns:
        labeled_df (Pandas DataFrame): dataframe with `tokenized snippet` (list of tokens) and `y`
            (financial flag)
    """
    labeled_data = []

    # get labeled data by looping through each row and extracting text snippets
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        for tag in row[tags_column_name]:
            row["clean_text"] = clean_text(row[text_column_name])
            text_snippets = extract_text_snippets(tokenized_text=row["clean_text"],
                                                  tag=tag,
                                                  window_size=window_size)
            for snippet in text_snippets:
                data_point = {"tokenized_snippet": snippet}

                # add outcome if dataset being used for training
                if training:
                    data_point["y"] = row[financial_flag_column_name] * 1

                labeled_data.append(data_point)

    labeled_df = pd.DataFrame(labeled_data)

    return labeled_df
