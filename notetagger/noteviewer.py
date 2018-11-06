import pandas as pd
from prompt_toolkit import prompt

from notetagger import constants


class NoteViewer:

    def __init__(self,
                 predictions_data,
                 original_data,
                 join_column_names,
                 text_column_name,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME,
                 validation_column_name=constants.OUTCOME_COLUMN_NAME,
                 threshold=0.5):

        self._predictions_data = predictions_data
        self.data = self._merge_dataset(predictions_data, original_data, join_column_names)
        self._prediction_column_name = prediction_column_name
        self._validation_column_name = validation_column_name
        self.threshold = threshold

    def _merge_dataset(self, predictions_data, original_data, join_columns):
        """
        Merge back in original dataset before tagging occured. Merge is done on `join_columns`

        Arguments:
            predictions_data (Pandas DataFrame): predictions made by the `NoteTagger` class.
            original_data (Pandas DataFrame): dataset used by `NoteTagger` class to make predictions
            join_columns (str or list of str): columns to make join the two dataframes on

        Returns:
            merged_data (Pandas DataFrame): pandas dataframe of merged notes
        """
        merged_data = predictions_data.merge(original_data,
                                             how='right',
                                             on=join_columns)

        return merged_data

    def quick_stats(self):
        """
        Display the percentage of notes that were tagged (`coverage`, may be less than 100% if `word_tags` were used)
        as well as the percentage of notes that were positively tagged
        """
        num_records = len(self.data.index)
        total_preds = self.data[self._prediction_column_name].notnull().sum()

        coverage = total_preds / num_records * 100
        print("Coverage: {0:.2f}%".format(coverage))

        positive_tags = (len(self.data[self.data[self._prediction_column_name] > self.threshold].index) /
                         total_preds) * 100
        print("Positive Tags: {0:.2f}%".format(positive_tags))

    def _validation_set_generator(self):
        if self._validation_column_name not in self._predictions_data.columns:
            self._predictions_data.columns[self._validation_column_name] = None

        validation_set = self._predictions_data[self.predictions_data[self._validation_column_name].isnull()]
        for index, row in validation_set.iterrows():
            yield row

    def validate_predictions(self):

        for note in self._validation_set_generator():
            user_input = ''
            while user_input not in ['y', 'n']:
                print(note)
                print()
                print()
                user_input = prompt('Note reflects tag (y/n)')


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_data_path',
                        '-p',
                        required=True,
                        type=str,
                        help='Path to predictions data must be jsonl')

    parser.add_argument('--original_data',
                        '-o',
                        required=True,
                        type=str,
                        help='Path to data on which predictions were made (contains the note text), must be jsonl')

    parser.add_argument('--join_columns',
                        '-j',
                        required=True,
                        action='append',
                        type=str,
                        help='Column on which to join the datasets on')

    parser.add_argument('--text_column_name',
                        '-t',
                        required=True,
                        type=str,
                        help='Name of text column')

    args = parser.parse_args()

    predictions_data = pd.read_json(args.predictions_data_path, orient='records', lines=True)
    original_data = pd.read_json(args.original_data_path, orient='records', lines=True)

    note_viewer = NoteViewer(predictions_data=predictions_data,
                             original_data=original_data,
                             join_column_names=args.join_columns,
                             text_column_name=args.text_column_name)

    note_viewer.validate_predictions()


if __name__ == '__main__':
    main()
