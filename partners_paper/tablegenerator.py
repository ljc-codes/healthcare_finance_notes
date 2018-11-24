import pandas as pd
from tabulate import tabulate

from notetagger import constants


class TableGenerator:

    def __init__(self,
                 predictions_filepath,
                 patient_data_filepath,
                 prediction_column_label=constants.PREDICTION_COLUMN_NAME,
                 predictions_threshold=0.8,
                 note_id_column='note_id',
                 patient_id_column='subject_id'):
        """
        Initializes the Table Generator Table used to produce tables for publication

        Arguments:
            predictions_filepath (str): path to jsonl file with model predictions
            patient_data_filepath (str): path to txt file with patient data that can be joined to
                `predictions_filepath` data

        Keyword Arguments:
            prediction_column_label (str): label of predictions column
            predictions_threshold (float): threshold for a positive financial note prediction
            note_id_column (str): label of column with the note ids
            patient_id_column (str): label of column with patient ids
        """

        self._patient_id_column = patient_id_column
        self._prediction_column_label = prediction_column_label

        # load data and merge together
        predictions = pd.read_json(predictions_filepath, orient='records', lines=True)
        patient_data = pd.read_csv(patient_data_filepath, sep='\t')
        self.notes_data = patient_data.merge(predictions[[note_id_column, prediction_column_label]],
                                             how='left',
                                             on=note_id_column)

        # format predictions column
        self.notes_data[prediction_column_label].fillna(0, inplace=True)
        self.notes_data[prediction_column_label] = (self.data[prediction_column_label] > predictions_threshold) * 1

        # get ids of those patients with a financial note
        financial_notes_indices = self.notes_data[self._prediction_column_label] == 1
        self.patient_ids = self.notes_data[financial_notes_indices][self._patient_id_column].unique()

    def _get_total_stats(self,
                         df):
        """
        Gets counts of total records and the number of financial notes in a dataframe, formatted for table
        output

        Arguments:
            df (Pandas DataFrame): dataframe with the `_prediction_column_label`

        Returns:
            stats (dict): dict with 'total' and 'financial' keys formatted for table output
        """
        total = df.shape[0]
        financial = df[self._prediction_column_label].sum()
        stats = {'total': '{0:,}'.format(total),
                 'financial': '{0:,} ({1:.2f})'.format(financial, financial / total * 100)}
        return stats

    def create_summary_table(self):
        """
        Prints a summary table for both patients and notes in terms of total records and financial records
        """

        # get stats for both patients and notes
        notes_stats = self._get_total_stats(self.notes_data)
        patients_stats = self._get_total_stats(
            self.notes_data[self.notes_data[self._patient_id_column].isin(self.patient_ids)]
            .drop_duplicates(self._patient_id_column))

        # print table
        print(tabulate([['Total', patients_stats['total'], notes_stats['total']],
                        ['Financial', patients_stats['financial'], notes_stats['financial']]],
                       headers=['', 'Patients', 'Notes']))
