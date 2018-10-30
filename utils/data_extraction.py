import psycopg2 as pg
import pandas as pd

from utils import constants


def extract_notes(db_config,
                  note_id_column_name=constants.NOTE_ID_COLUMN_NAME):
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
    SELECT flagged_notes.row_id as {0}, flagged_notes.text, flagged_notes.financial_flag, flagged_notes._cost__tag,
           flagged_notes.insurance_tag, flagged_notes.pay_tag, flagged_notes.financial_tag, flagged_notes.expense_tag,
           flagged_notes.afford_tag, flagged_notes.medicare_tag, flagged_notes.out_of_pocket_tag,
           flagged_notes.expensive_tag,flagged_notes.costly_tag
    FROM flagged_notes
    WHERE flagged_notes.iserror is null
    """.format(note_id_column_name)

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
