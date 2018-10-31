import os

import psycopg2 as pg
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import constants
from utils.text_processing import process_text


def extract_notes(db_config,
                  note_id_column_name=constants.NOTE_ID_COLUMN_NAME,
                  text_column_name=constants.TEXT_COLUMN_NAME,
                  financial_flag_column_name=constants.FINANCIAL_FLAG_COLUMN_NAME):
    """
    Extract notes from postgres db into pandas dataframe

    Arguments:
        db_config (str): config for postgres db

    Returns:
        df (Pandas DataFrame): Pandas dataframe with note id, note text, and list of string tags
    """
    db_connection = pg.connect(db_config)

    # exclude any notes tagged as `is_error`
    query = """
    SELECT flagged_notes.row_id as {0}, flagged_notes.text as {1}, flagged_notes.financial_flag as {2},
           flagged_notes._cost__tag, flagged_notes.insurance_tag, flagged_notes.pay_tag, flagged_notes.financial_tag,
           flagged_notes.expense_tag, flagged_notes.afford_tag, flagged_notes.medicare_tag,
           flagged_notes.out_of_pocket_tag, flagged_notes.expensive_tag,flagged_notes.costly_tag
    FROM flagged_notes
    WHERE flagged_notes.iserror is null
    """.format(note_id_column_name, text_column_name, financial_flag_column_name)

    df = pd.read_sql(query, db_connection)
    df = format_dataframe(df)
    return df


def format_dataframe(df,
                     tags_column_name=constants.TAGS_COLUMN_NAME):
    """
    Converts tag flags to list of strings

    Arguments:
        df (Pandas DataFrame): Pandas dataframe with note id, note text, and various flags

    Returns:
        df (Pandas DataFrame): Pandas dataframe with note id, note text, and list of string tags
    """
    # get tag and non tag columns
    tag_columns = [col for col in df.columns if "tag" in col]
    non_tag_columns = [col for col in df.columns if "tag" not in col]

    df[tags_column_name] = df.apply(lambda row: [col.replace("_tag", "").replace("_", "")
                                                 for col in tag_columns if row[col]], axis=1)

    df = df[non_tag_columns + [tags_column_name]]
    return df


def split_df(df,
             test_size=.10,
             random_state=42):
    """
    Split a dataframe into a train and test set

    Arguments:
        df (Pandas Dataframe): dataframe with feature and outcome labels

    Keyword Arguments:
        test_size (float): proportion of dataset to use as test set
        random_state (int): seed for splitting that allows for replication of results

    Returns:
        train_df (Pandas Dataframe): dataframe for training
        test_df (Pandas Dataframe): dataframe for testing
    """
    train_df, test_df = train_test_split(df,
                                         test_size=test_size,
                                         random_state=random_state)
    return train_df, test_df


def download_train_test_set(window_size,
                            save_folder):
    """
    Downloads and saves training and test datasets

    Arguments:
        window_size (int): number of tokens to get both before and after the `tag`
        save_folder (str): folder to save files to
    """
    df = extract_notes(os.environ["DB_CONFIG"])
    df_clean = process_text(df=df,
                            window_size=window_size,
                            training=True)
    train_df, test_df = split_df(df_clean)

    # if save folder does not exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save train_df
    train_filepath = os.path.join(save_folder, "training_ws{0}.jsonl".format(window_size))
    train_df.to_json(train_filepath,
                     orient='records',
                     lines=True)

    # save test_df
    test_filepath = os.path.join(save_folder, "testing_ws{0}.jsonl".format(window_size))
    test_df.to_json(test_filepath,
                    orient='records',
                    lines=True)
