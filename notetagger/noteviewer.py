import pandas as pd
import numpy as np
from prompt_toolkit import prompt
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from notetagger import constants


class NoteViewer:

    def __init__(self,
                 predictions_data,
                 original_data,
                 join_column_names,
                 text_column_name,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME,
                 validation_column_name=constants.VALIDATION_COLUMN_NAME,
                 word_tags=constants.TAGS,
                 text_window_before=300,
                 text_window_after=100,
                 text_window_increment=200,
                 threshold=0.5):

        self._prediction_data_columns = list(predictions_data.columns)
        self.data = self._merge_dataset(predictions_data, original_data, join_column_names)
        self._text_column_name = text_column_name
        self._prediction_column_name = prediction_column_name
        self._validation_column_name = validation_column_name
        self._word_tags = word_tags
        self._text_window_before = text_window_before
        self._text_window_after = text_window_after
        self._text_window_increment = text_window_increment
        self.threshold = threshold

    def _merge_dataset(self, predictions_data, original_data, join_column_names):
        """
        Merge back in original dataset before tagging occured. Merge is done on `join_columns` and then
        data is shuffled to eliminate bias from validation

        Arguments:
            predictions_data (Pandas DataFrame): dataset of predictions output by `NoteTagger` class
            original_data (Pandas DataFrame): dataset used by `NoteTagger` class to make predictions
            join_column_names (list of str): columns to join dataset with

        Returns:
            merged_data (Pandas DataFrame): pandas dataframe of merged notes
        """
        merged_data = predictions_data.merge(original_data,
                                             how='right',
                                             on=join_column_names)

        # shuffle data
        merged_data = merged_data.reindex(np.random.permutation(merged_data.index))

        return merged_data

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

        positive_tags = (len(self.data[self.data[self._prediction_column_name] > self.threshold].index) /
                         total_preds) * 100
        print("Prediction Positive Tags: {0:.2f}%".format(positive_tags))

        validation_coverage = total_validation / total_preds * 100
        print("Validation Coverage: {0:.2f}%".format(validation_coverage))

        comparison_set = self.data[self.data[self._validation_column_name].notnull()]
        y_true = comparison_set[self._validation_column_name]
        y_pred = comparison_set[self._prediction_column_name]
        print("AUC: {:.2f}\n".format(roc_auc_score(y_true=y_true, y_score=y_pred)))

        # print metrics at various thresholds
        thresholds = [i / 10 for i in range(4, 10)]
        for threshold in thresholds:
            print("Threshold: {:.1f}\n{}".format(threshold, '-' * 12))
            print("Accuracy: {:.2f}".format(accuracy_score(y_true=y_true, y_pred=y_pred > threshold)))
            print("Precision: {:.2f}".format(precision_score(y_true=y_true, y_pred=y_pred > threshold)))
            print("Recall: {:.2f}\n".format(recall_score(y_true=y_true, y_pred=y_pred > threshold)))

    def _validation_set_generator(self):
        """
        Generator that yields notes that need to be validated

        Yields:
            index (int): index of the specific record
            row (dict): dict of data for a particular record
        """
        if self._validation_column_name not in self.data.columns:
            self.data[self._validation_column_name] = None

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
                word_tag_index = full_note_text.find(word_tag)
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

        # initialize columns and indices to use when saving down data
        validation_data_columns = self._prediction_data_columns + [self._validation_column_name]
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
                        self.data[validation_data_filter_index][self._validation_column_name].notnull().sum() + 1
                    )
                    pct_records_validated = records_validated / self.data[validation_data_filter_index].shape[0] * 100
                    print('{} records validated, {:.0f}% of total records'.format(records_validated,
                                                                                  pct_records_validated))

                    # save data to disk
                    self.data[validation_data_filter_index][validation_data_columns].to_json(validation_save_path,
                                                                                             orient='records',
                                                                                             lines=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--predictions_data_path',
                        '-p',
                        required=True,
                        type=str,
                        help='Path to predictions data must be jsonl')

    parser.add_argument('--original_data_path',
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

    note_viewer.validate_predictions(validation_save_path=args.predictions_data_path)


if __name__ == '__main__':
    main()
