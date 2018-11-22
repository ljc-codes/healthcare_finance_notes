import os

import psycopg2 as pg
import pandas as pd
from sklearn.model_selection import train_test_split

from notetagger import constants


def extract_notes(db_config,
                  note_id_column_name=constants.NOTE_ID_COLUMN_NAME,
                  text_column_name=constants.TEXT_COLUMN_NAME,
                  financial_flag_column_name=constants.OUTCOME_COLUMN_NAME):
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
    SELECT flagged_notes.row_id as {0}, flagged_notes.text as {1}, flagged_notes.financial_flag as {2}
    FROM flagged_notes
    WHERE flagged_notes.iserror is null
    """.format(note_id_column_name, text_column_name, financial_flag_column_name)

    df = pd.read_sql(query, db_connection)
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


def save_data(df, save_folder, filename):
    """
    Saves data to a jsonl files

    Arguments:
        df (Pandas Dataframe): data to save
        save_folder (str): folder to save data in
        filename (str): name of file
    """
    filepath = os.path.join(save_folder, filename)
    df.to_json(filepath,
               orient='records',
               lines=True)


def download_train_test_set(save_folder):
    """
    Downloads and saves training and test datasets

    Arguments:
        save_folder (str): folder to save files to
    """
    df = extract_notes(os.environ["DB_CONFIG"])
    train_df, test_df = split_df(df)

    # if save folder does not exist, create it
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save train_df
    save_data(train_df, save_folder, "training_mimic.jsonl")

    # save test_df
    save_data(test_df, save_folder, "testing_mimic.jsonl")


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_folder',
                        '-s',
                        default=os.getcwd().replace('training', 'data'),
                        type=str,
                        help='folder to save files to')

    args = parser.parse_args()

    download_train_test_set(save_folder=args.save_folder)
