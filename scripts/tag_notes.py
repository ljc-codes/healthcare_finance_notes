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
        Loads data with raw note text (must be jsonl), tag it with a given a model, and save the predictions
        and any metadata columns to anoter jsonl file

        Arguments:
            model_path (str): filepath to a NoteTaggerTrainedModel
            input_data_path (str): filepath to data with raw note text
            output_data_path (str): filepath to save prediction data to
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


def main():
    """
    Command line handler for the script
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_path',
                        '-m',
                        required=True,
                        type=str,
                        help='filepath to model')

    parser.add_argument('--input_data_path',
                        '-i',
                        required=True,
                        type=str,
                        help='filepath to input data, should be jsonl')

    parser.add_argument('--output_data_path',
                        '-o',
                        required=True,
                        type=str,
                        help='filepath to output data, should be jsonl')

    parser.add_argument('--text_column_name',
                        '-t',
                        required=True,
                        type=str,
                        help='name of column of with raw text in data')

    parser.add_argument('--metadata_columns',
                        '-md',
                        action='append',
                        type=str,
                        help='columns to include with predictions')

    parser.add_argument('--prediction_column_name',
                        '-p',
                        default=constants.PREDICTION_COLUMN_NAME,
                        type=str,
                        help='name of column with predictions')

    args = parser.parse_args()

    tag_notes(model_path=args.model_path,
              input_data_path=args.input_data_path,
              output_data_path=args.output_data_path,
              text_column_name=args.text_column_name,
              metadata_columns=args.metadata_columns,
              prediction_column_name=args.prediction_column_name)
