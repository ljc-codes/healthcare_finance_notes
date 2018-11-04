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

    # create token for out of pocket
    text_oop = text.replace('out of pocket', 'outofpocket')

    tokenized_text = nltk.word_tokenize(text_oop)
    lower_case_words = [word.lower() for word in tokenized_text if word.isalpha()]
    clean_text = [word for word in lower_case_words if word not in STOPWORDS]
    return clean_text


def extract_text_snippets_with_tags(tokenized_text, tags, window_size):
    """
    Gets multiple lists of tokens of window size * 2 surrounding tag words

    Arguments:
        tokenized_text (list of str): list of tokens
        tags (list of str): words to get window around
        window_size (int): number of tokens to get both before and after the `tag`

    Returns:
        text_snippets (list of list of str): list of list of tokens of size window * 2
    """

    tag_indices = [i for i, token in enumerate(tokenized_text) if sum([1 for tag in tags if tag in token]) > 0]

    window_params = [(max(0, index - window_size), min(len(tokenized_text), index + window_size))
                     for index in tag_indices]

    text_snippets = [tokenized_text[param[0]:param[1]] for param in window_params]
    return text_snippets


def extract_text_snippets(tokenized_text,
                          window_size,
                          stride_length):
    """
    Get multiple lists of tokens of window size * 2 by moving the window across the entire text

    Arguments:
        tokenized_text (list of str): list of tokens
        window_size (int): number of tokens to get both before and after the `tag`
        stride_length (int): number of tokens to move window each time

    Returns:
        text_snippets (list of list of str): list of list of tokens of size window * 2
    """
    token_length = max(len(tokenized_text) - window_size * 2 + 1,
                       1)
    text_snippets = [tokenized_text[i:i + window_size * 2]
                     for i in range(0, token_length, stride_length)]
    return text_snippets


def process_text(df,
                 window_size,
                 tags=constants.TAGS,
                 stride_length=50,
                 text_column_name=constants.TEXT_COLUMN_NAME,
                 columns_to_keep=[constants.NOTE_ID_COLUMN_NAME, constants.OUTCOME_COLUMN_NAME],
                 feature_column_name=constants.FEATURE_COLUMN_NAME):
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

        row["clean_text"] = clean_text(row[text_column_name])

        if tags:
            # if creating snippets around specific tags, extract snippets for each tag and then flatten the list
            text_snippets = extract_text_snippets_with_tags(tokenized_text=row["clean_text"],
                                                            window_size=window_size,
                                                            tags=tags)
        else:
            # if not using tags, extract sliding window over text
            text_snippets = extract_text_snippets(tokenized_text=row["clean_text"],
                                                  window_size=window_size,
                                                  stride_length=stride_length)

        # loop through each snippet and add it to a dict along with any other chosen columns
        for snippet in text_snippets:
            data_point = {feature_column_name: snippet}

            for column in columns_to_keep:
                data_point[column] = row[column]

            labeled_data.append(data_point)

    labeled_df = pd.DataFrame(labeled_data)

    return labeled_df
