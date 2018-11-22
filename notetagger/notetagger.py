import dill as pickle
import pandas as pd
import numpy as np
from prompt_toolkit import prompt

from notetagger import constants
from notetagger import metrics_calculation


class NoteTagger:
    """
    This class provides a command line interface to manually tag notes and compare them to predictions
    made by a model

    # initialize class
    tagger = NoteTagger(predictions_data=PandasDataframe,
                        original_data=PandasDataframe,
                        metadata_columns=[str,...],
                        text_column_name=str)

    # start validator
    tagger.validate_predictions(validation_save_path=str)
    """

    def __init__(self,
                 original_data,
                 predictions_data,
                 text_column_name,
                 metadata_columns,
                 model_path=None,
                 word_tags=constants.TAGS,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME,
                 validation_column_name=constants.VALIDATION_COLUMN_NAME,
                 text_window_before=300,
                 text_window_after=10,
                 text_window_increment=200):
        """
        Initialize NoteViewer Object

        Arguments:
            original_data (Pandas DataFrame): DataFrame that was fed into a `NoteTaggerTrainedModel`
                to produce predictions_data
            predictions_data (Pandas DataFrame): DataFrame containing model predictions on notes,
                output by the `NoteTaggerTrainedModel`. If `None`, this will be calculated at initialization
            text_column_name (str): name column with raw note text
            metadata_columns (list of str): list of column names to include with `predictions_data`. Used
                to join `predictions_data` and `original_data`

        Keyword Arguments:
            model_path (str): path to model, only used if `predictions_data` is `None`
            prediction_column_name (str): name of column with model's predictions
            validation_column_name (str): name of column that user's tags are saved to
            word_tags (list of str): list of tags which the model predicted on. If `word_tags` were not used in
                model predictions, should be set to None
            text_window_before (int): number of characters to print before word tag
            text_window_after (int): number of characters to print after tag
            text_window_increment (int): number of characters to increase window by if user selects option
        """
        self._text_column_name = text_column_name
        self._metadata_columns = metadata_columns
        self._model_path = model_path
        self._word_tags = word_tags
        self._prediction_column_name = prediction_column_name
        self._validation_column_name = validation_column_name
        self._text_window_before = text_window_before
        self._text_window_after = text_window_after
        self._text_window_increment = text_window_increment

        self._create_dataset(original_data, predictions_data)

    def _create_dataset(self, original_data, predictions_data):
        """
        Merge back in original dataset before tagging occured. Merge is done on `join_columns` and then
        data is shuffled to eliminate bias from validation.

        Arguments:
            predictions_data (Pandas DataFrame): dataset of predictions output by `NoteTaggerTrainedModel`.
                If `None`, the model is loaded and predictions are made
            original_data (Pandas DataFrame): dataset used by `NoteTaggerTrainedModel` to make predictions
        """

        # check if predictions data is provided
        if predictions_data is None:

            # load model
            with open(self._model_path, 'rb') as f:
                model = pickle.load(f)

            # create predictions dataset
            predictions_data = model.predict_tag(
                data=original_data,
                text_column_name=self._text_column_name,
                metadata_columns=self._metadata_columns,
                prediction_column_name=self._prediction_column_name)

        # get columns unique to prediction data, used for saving it down after user tags
        self._prediction_data_columns = list(predictions_data.columns)

        # merge predictions data with original data
        self._dataset = predictions_data.merge(original_data,
                                               how='right',
                                               on=self._metadata_columns)

        # shuffle data
        self._dataset = self._dataset.reindex(np.random.permutation(self._dataset.index))

    def quick_stats(self):
        """
        Display the percentage of notes that were tagged (`coverage`, may be less than 100% if `word_tags` were used)
        as well as the percentage of notes that were positively tagged
        """
        num_records = len(self.data.index)
        total_preds = self.data[self._prediction_column_name].notnull().sum()
        total_validation = self.data[self._validation_column_name].notnull().sum()

        prediction_coverage = total_preds / num_records * 100
        print("Prediction Coverage: {0:.2f}%".format(prediction_coverage))

        validation_coverage = total_validation / total_preds * 100
        print("Validation Coverage: {0:.2f}%".format(validation_coverage))

        # print performance metrics at various thresholds
        comparison_set = self.data[self.data[self._validation_column_name].notnull()]
        y_true = comparison_set[self._validation_column_name].astype('float64')
        y_pred = comparison_set[self._prediction_column_name]
        print(metrics_calculation.calculate_performance_metrics(y_true, y_pred))

    def _validation_set_generator(self):
        """
        Generator that yields notes that need to be validated

        Yields:
            index (int): index of the specific record
            row (dict): dict of data for a particular record
        """

        # create columns to keep when saving down validation data
        self._validation_data_columns = self._prediction_data_columns

        # if data has not been validated before, create a validation column
        if self._validation_column_name not in self.data.columns:
            self.data[self._validation_column_name] = None
            self._validation_column_name = self._validation_data_columns + [self._validation_column_name]

        # validate those records which have a prediction but no validation
        validation_set_indices = (
            self.data[self._validation_column_name].isnull() & self.data[self._prediction_column_name].notnull()
        )
        validation_set = self.data[validation_set_indices]

        for index, row in validation_set.iterrows():
            yield index, row

    def _print_note_text(self,
                         full_note_text,
                         word_tags,
                         text_window_before,
                         text_window_after):
        """
        Prints note text either a specific window around a word if given word tags or the entire text if not

        Arguments:
            full_note_text (str): entire text of note
            word_tags (list of str): tags to find in notes and then print a window around. If None, the entire
                note is printed
            text_window_before (int): number of characters to print before tag
            text_window_after (int): number of characteres to print after tag
        """
        if word_tags:
            for word_tag in word_tags:
                word_tag_index = full_note_text.lower().find(word_tag.replace('_', ' '))
                if word_tag_index >= 0:
                    print('\ntag: {}\n{}\n'.format(word_tag, len('tag: ' + word_tag) * '-'))
                    text_snippet = full_note_text[max(0, word_tag_index - text_window_before):
                                                  min(len(full_note_text),
                                                      word_tag_index + text_window_after)]
                    print('{}\n\n'.format(text_snippet))
        else:
            print(full_note_text)

    def validate_predictions(self, validation_save_path):
        """
        Iterates through notes that have predictions but have not been validated by a person,
        displaying a dialog box for the person to validate these notes. Each time a note is validated
        the data (only those rows with predictions) is saved down

        Arguments:
            validation_save_path (str): path to save data to, should be jsonl
        """

        # initialize vars to handle user inputs
        valid_inputs = ['y', 'n', 'q']
        prompt_text = ("-----------------------------------\n"
                       "Please choose one of the following:\n"
                       "(y) flag note\n"
                       "(n) don't flag note\n"
                       "(q) quit validator\n")

        # add some inputs if using word tags to find relevant text
        if self._word_tags:
            valid_inputs.extend(['b', 'a'])
            additional_prompt_text = ("(b) increase window size before text\n"
                                      "(a) increase window after text\n")
            prompt_text += additional_prompt_text

        # initialize indices to use when saving down data
        validation_data_filter_index = self.data[self._prediction_column_name].notnull()

        # iterate through each record that needse to be validated
        for index, note in self._validation_set_generator():

            # initialize variables for individual note
            text_window_before = self._text_window_before
            text_window_after = self._text_window_after
            user_input = ''

            # keep displaying the note until the user enters a valid flag
            while user_input not in ['y', 'n']:

                # print note text
                self._print_note_text(full_note_text=note[self._text_column_name],
                                      word_tags=self._word_tags,
                                      text_window_before=text_window_before,
                                      text_window_after=text_window_after)

                # print user prompt
                user_input = prompt(prompt_text)

                # handle user input
                if user_input not in valid_inputs:
                    print("\nPlease enter a valid input!")
                    continue
                elif user_input == 'q':
                    print('\nQuitting Validator...')
                    self.quick_stats()
                    return
                elif user_input == 'b':
                    print('\nIncreasing text window before tag...')
                    text_window_before += self._text_window_increment
                elif user_input == 'a':
                    print('\nIncreasing text window after tag...')
                    text_window_after += self._text_window_increment
                else:
                    # user has entered a valid flag
                    print('\nSaving validation flag...\n')

                    # save flag to dataframe, converting to a boolean
                    self.data.loc[index, self._validation_column_name] = (user_input == 'y')

                    # print number of records validated
                    records_validated = (
                        self.data[validation_data_filter_index][self._validation_column_name].notnull().sum()
                    )
                    pct_records_validated = records_validated / self.data[validation_data_filter_index].shape[0] * 100
                    print('{} records validated, {:.0f}% of total records'.format(records_validated,
                                                                                  pct_records_validated))

                    # save data to disk
                    self.data[validation_data_filter_index][self._validation_data_columns].to_json(validation_save_path,
                                                                                                   orient='records',
                                                                                                   lines=True)


def main():
    """
    Command line handler for the script. Includes handling for loading jsonl data and saving jsonl data
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_data_path',
                        '-p',
                        default=None,
                        type=str,
                        help='Path to predictions data must be jsonl')

    parser.add_argument('--original_data_path',
                        '-o',
                        required=True,
                        type=str,
                        help='Path to data on which predictions were made (contains the note text), must be jsonl')

    parser.add_argument('--metadata_columns',
                        '-md',
                        required=True,
                        action='append',
                        type=str,
                        help='Column(s) on which to join the datasets on')

    parser.add_argument('--text_column_name',
                        '-t',
                        required=True,
                        type=str,
                        help='Name of text column')

    parser.add_argument('--prediction_column_name',
                        '-pc',
                        default=constants.PREDICTION_COLUMN_NAME,
                        type=str,
                        help='Name of model prediction column')

    parser.add_argument('--validation_column_name',
                        '-vc',
                        default=constants.VALIDATION_COLUMN_NAME,
                        type=str,
                        help='Name of user validation inputs column')

    parser.add_argument('--word_tags',
                        '-w',
                        default=constants.TAGS,
                        action='append',
                        type=str,
                        help='Tags on which the model predicted on')

    parser.add_argument('--text_window_before',
                        '-tb',
                        default=300,
                        type=int,
                        help='Number of characters to print before word tag')

    parser.add_argument('--text_window_after',
                        '-ta',
                        default=10,
                        type=int,
                        help='Number of characters to print after word tag')

    parser.add_argument('--text_window_increment',
                        '-ti',
                        default=200,
                        type=int,
                        help='Number of characters to increase window by if user selects option')

    parser.add_argument('--model_path',
                        '-m',
                        default=None,
                        type=str,
                        help='Path to model to use to make predictions data')

    args = parser.parse_args()

    predictions_data = pd.read_json(args.predictions_data_path, orient='records', lines=True)
    original_data = pd.read_json(args.original_data_path, orient='records', lines=True)

    note_viewer = NoteTagger(predictions_data=predictions_data,
                             original_data=original_data,
                             metadata_columns=args.metadata_columns,
                             text_column_name=args.text_column_name,
                             prediction_column_name=args.prediction_column_name,
                             validation_column_name=args.validation_column_name,
                             word_tags=args.word_tags,
                             text_window_before=args.text_window_before,
                             text_window_after=args.text_window_after,
                             text_window_increment=args.text_window_increment,
                             model_path=args.model_path)

    note_viewer.validate_predictions(validation_save_path=args.predictions_data_path)
