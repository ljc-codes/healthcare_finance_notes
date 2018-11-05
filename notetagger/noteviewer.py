from notetagger import constants


class NoteViewer:

    def __init__(self,
                 predictions_data,
                 original_data,
                 join_column_names,
                 text_column_name,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME,
                 threshold=0.5):

        self.data = self._merge_dataset(predictions_data, original_data, join_column_names)
        self._prediction_column_name = prediction_column_name
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

        coverage = self.data[self._prediction_column_name].isnull().sum() / num_records * 100
        print("Coverage: {0:.2f}%".format(coverage))

        positive_tags = (len(self.data[self.data[self._prediction_column_name] > self.threshold].index) /
                         num_records) * 100
        print("Positive Tags: {0:.2f}%".format(positive_tags))
