import pandas as pd
import dill as pickle

from notetagger import constants


def tag_notes(model_path,
              input_data_path,
              output_data_path,
              text_column_name,
              metadata_columns,
              prediction_column_name=constants.PREDICTION_COLUMN_NAME):
        """
        Arguments:

            text_column_name (str): label of column that contains raw note text
            metadata_columns (list of str): any column names to include with note tag predictions. Must be a list

        Keyword Arguments:
            prediction_column_name (str): label of column that contains the predicted tag
        """

        print("Loading model from {}".format(model_path))
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        print("Loading data from {}".format(input_data_path))
        input_data = pd.read_json(input_data_path, orient='records', lines=True)

        print("Making predictions")
        note_tag_predictions = model.predict_tag(
            data=input_data,
            text_column_name=text_column_name,
            metadata_columns=metadata_columns,
            prediction_column_name=prediction_column_name)

        print("Saving data to {}".format(output_data_path))
        note_tag_predictions.to_json(output_data_path, orient='records', lines=True)
