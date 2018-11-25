import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from tabulate import tabulate

from notetagger import constants


class TableGenerator:

    def __init__(self,
                 predictions_filepath,
                 patient_data_filepath,
                 prediction_column=constants.PREDICTION_COLUMN_NAME,
                 predictions_threshold=0.8,
                 note_id_column='note_id',
                 note_date_column='note_date',
                 patient_id_column='subject_num',
                 categorical_columns=['gender', 'race']):
        """
        Initializes the Table Generator Table used to produce tables for publication

        Arguments:
            predictions_filepath (str): path to jsonl file with model predictions
            patient_data_filepath (str): path to txt file with patient data that can be joined to
                `predictions_filepath` data

        Keyword Arguments:
            prediction_column (str): label of predictions column
            predictions_threshold (float): threshold for a positive financial note prediction
            note_id_column (str): label of column with the note ids
            note_date_column (str): label of column with the note dates
            patient_id_column (str): label of column with patient ids
        """

        self._prediction_column = prediction_column
        self._patient_id_column = patient_id_column
        self._categorical_columns = categorical_columns

        # load data and merge together
        predictions = pd.read_json(predictions_filepath, orient='records', lines=True)
        patient_data = pd.read_csv(patient_data_filepath, sep='\t', parse_dates=[note_date_column])
        self.notes_data = patient_data.merge(predictions[[note_id_column, prediction_column]],
                                             how='left',
                                             on=note_id_column)

        # format predictions column
        self.notes_data[prediction_column].fillna(0, inplace=True)
        self.notes_data[prediction_column] = (self.notes_data[prediction_column] > predictions_threshold)

        # get ids of those patients with a financial note
        financial_notes_indices = self.notes_data[self._prediction_column] == 1
        patient_ids = self.notes_data[financial_notes_indices][self._patient_id_column].unique()

        # separate data into those patients w/ tags and those patients w/out tags, keeping the first patient visit
        self.patients_w_tags = (self.notes_data[self.notes_data[self._patient_id_column].isin(patient_ids)]
                                .sort_values(note_date_column)
                                .drop_duplicates(self._patient_id_column))
        self.patients_wout_tags = (self.notes_data[~self.notes_data[self._patient_id_column].isin(patient_ids)]
                                   .sort_values(note_date_column)
                                   .drop_duplicates(self._patient_id_column))

    def _get_total_stats(self,
                         df):
        """
        Gets counts of total records and the number of financial notes in a dataframe, formatted for table
        output

        Arguments:
            df (Pandas DataFrame): dataframe with the `_prediction_column`

        Returns:
            stats (dict): dict with 'total' and 'financial' keys formatted for table output
        """
        total = df.shape[0]
        financial = df[self._prediction_column].sum()
        stats = {'total': '{0:,}'.format(total),
                 'financial': '{0:,} ({1:.2f}%)'.format(financial, financial / total * 100)}
        return stats

    def create_summary_table(self):
        """
        Prints a summary table for both patients and notes in terms of total records and financial records
        """

        # get stats for both patients and notes
        notes_stats = self._get_total_stats(self.notes_data)
        patients_stats = self._get_total_stats(
            self.notes_data
            .sort_values(self._prediction_column, ascending=False)
            .drop_duplicates(self._patient_id_column))

        # print table
        print(tabulate([['Total', patients_stats['total'], notes_stats['total']],
                        ['Financial', patients_stats['financial'], notes_stats['financial']]],
                       headers=['', 'Patients', 'Notes']))

    def _calc_chi2_counts(self, df, column_label, column_value):
        """
        Get counts of the desired value and all other records less the value for use in `_calc_chi2_test`

        Arguments:
            df (Pandas DataFrame): dataframe with the column label
            column_label (str): label of column which contains the value being tested
            column_value: value for which the chi2 is calculated

        Returns:
            f_count (list of int): count of both positive and negative instances of the `column_value`
        """
        pos_count = df[df[column_label] == column_value].shape[0]
        neg_count = df.shape[0] - pos_count
        f_count = [pos_count, neg_count]
        return f_count

    def _calc_chi2_test(self, column_label, column_value):
        """
        Performs a chi2 test for a specific categorical value

        Arguments:
            column_label (str): label of column which contains the value being tested
            column_value: value for which the chi2 is calculated

        Returns:
            chi2_data (list): list with the value of the column, record counts for both finance and non-finance,
                the chi2 statistic, and its p-value
        """

        # format data for test
        f_obs = self._calc_chi2_counts(self.patients_w_tags, column_label, column_value)
        f_exp = self._calc_chi2_counts(self.patients_wout_tags, column_label, column_value)

        # run chi2 test
        chi2_test = chi2_contingency(np.array([f_obs, f_exp]))

        # create response json
        chi2_data = [column_value,
                     '{0:,} ({1:.2f})'.format(f_obs[0], f_obs[0] / sum(f_obs) * 100),
                     '{0:,} ({1:.2f})'.format(f_exp[0], f_exp[0] / sum(f_exp) * 100),
                     chi2_test[0],
                     chi2_test[1]]
        return chi2_data

    def create_categorical_table(self):
        """
        Prints a summary table for chi2 comparisons of categorical values at the patient level
        """

        # loop through each categorical column and then each value in the column, adding the test data to the list
        categorical_table_data = []
        for column_label in self._categorical_columns:
            column_values = list(self.notes_data[column_label].unique())

            # loop through each column value in the column
            for column_value in column_values:
                categorical_table_data.append(
                    self._calc_chi2_test(column_label=column_label, column_value=column_value))

        print(tabulate(categorical_table_data, headers=['Feature', 'n (%)', 'n (%)', 'Chi Square', 'P-Value']))
