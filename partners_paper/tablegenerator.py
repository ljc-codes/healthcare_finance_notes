import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency
from scipy.stats import ttest_1samp
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
                 categorical_columns=['gender', 'race', 'marital_status', 'InsuranceType'],
                 numerical_columns=['age_at_visit', 'zip_median_income'],
                 features_to_exclude=['gender_M',
                                      'gender_U',
                                      'race_White',
                                      'marital_status_MARRIED',
                                      'InsuranceType_Private']):
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
            categorical_columns (list of str): list of columns to calculate categorical comparisons for
            numerical_columns (list of str): list of columns to calculate numerical comparisons for
            features_to_exclude (list of str): list of features to exclude from logistic regression
        """

        self._prediction_column = prediction_column
        self._note_id_column = note_id_column
        self._patient_id_column = patient_id_column
        self._categorical_columns = categorical_columns
        self._numerical_columns = numerical_columns
        self._features_to_exclude = features_to_exclude

        # load data and merge together
        predictions = pd.read_json(predictions_filepath, orient='records', lines=True)
        patient_data = pd.read_csv(patient_data_filepath, sep='\t', parse_dates=[note_date_column])
        self.notes_data = patient_data.merge(predictions[[self._note_id_column, prediction_column]],
                                             how='left',
                                             on=self._note_id_column)

        # clean categorical columns
        self._clean_demographic()

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

    def _clean_demographic(self,
                           zip_median_income_column='zip_median_income',
                           zip_median_income_scale=1000,
                           insurance_type_column='InsuranceType',
                           marital_status_column='marital_status'):
        """
        Cleans categorical columns for analysis

        Keyword Arguments:
            insurance_type_column (str): column name of insurance type
            marital_status_column (str): column name of marital status
        """
        self.notes_data[zip_median_income_column] = self.notes_data[zip_median_income_column] / zip_median_income_scale
        self.notes_data[insurance_type_column] = self.notes_data[insurance_type_column].fillna('Other/Unknown')
        self.notes_data[marital_status_column] = self.notes_data[marital_status_column].map(
            lambda x: "Other/Unknown" if x not in ["MARRIED", "SINGLE", "WIDOW", "DIVORCED"] else x)

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

    def create_population_stats_table(self):
        """
        Prints a summary table for mean / std for notes per patient and age, and overall distribution of sex and race
        """

        # get note counts per patient
        self._note_counts = self.notes_data.groupby(self._patient_id_column)[self._note_id_column].count().reset_index()
        self._note_counts.columns = [self._patient_id_column, 'note_count']

        # get stats for both patients and notes
        patient_stats = (self.notes_data.sort_values(self._prediction_column, ascending=False)
                    .drop_duplicates(self._patient_id_column))

        print(tabulate([['Notes per Patient',
                         '{:.3f}'.format(self._note_counts['note_count'].mean()),
                         '{:.3f}'.format(self._note_counts['note_count'].std())],
                       ['Age',
                        '{:.3f}'.format(patient_stats['age_at_visit'].mean()),
                        '{:.3f}'.format(patient_stats['age_at_visit'].std())]],
                       headers=['', 'Mean', 'Std']))

        gender_value_counts = patient_stats['gender'].value_counts() / patient_stats.shape[0]
        table_data = [[label, '{:.2%}'.format(value)]
                      for label, value in zip(gender_value_counts.index.tolist(),
                                              gender_value_counts.tolist())]
        table_data.append(['', ''])

        race_value_counts = patient_stats['race'].value_counts() / patient_stats.shape[0]
        table_data.extend([[label, '{:.2%}'.format(value)]
                           for label, value in zip(race_value_counts.index.tolist(),
                                                   race_value_counts.tolist())])
        print(tabulate(table_data, headers=['', '% of Total']))

    def _format_p_value(self, p_value, num_comparisons=1):
        """
        Get formatted p-value for displaying in table

        Arguments:
            p_value (float): p_value output by some statistical test

        Keyword Arguments:
            num_comparisons (int): number of comparisons to use for bonferroni correction

        Returns:
            formatted_p_value (str): stars associated with p-value
        """
        formatted_p_value = '<0.001' if p_value < 0.001 else '{0:.3f}'.format(p_value)
        return formatted_p_value

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
                     '{0:,} ({1:.2%} / {2:.2%})'.format(f_obs[0],
                                                        f_obs[0] / (f_obs[0] + f_exp[0]),
                                                        f_obs[1] / (f_obs[1] + f_exp[1])),
                     '{0:,} ({1:.2%})'.format(f_exp[0],
                                              f_exp[0] / (f_obs[0] + f_exp[0])),
                     '{0:.2f}'.format(chi2_test[0]),
                     self._format_p_value(chi2_test[1])]
        return chi2_data

    def create_categorical_table(self):
        """
        Prints a summary table for chi2 comparisons of categorical values at the patient level
        """

        # loop through each categorical column and then each value in the column, adding the test data to the list
        categorical_table_data = []
        for column_label in self._categorical_columns:

            # add header for column label in table
            categorical_table_data.append([column_label.capitalize(), '', '', '', ''])

            # get all column values in column
            column_values = list(self.notes_data[column_label].unique())

            # loop through each column value in the column
            for column_value in column_values:
                categorical_table_data.append(
                    self._calc_chi2_test(column_label=column_label, column_value=column_value))

        print(tabulate(categorical_table_data, headers=['Feature', 'n (%)', 'n (%)', 'Chi Square', 'P-Value']))

    def _calc_t_test_stats(self, df, column_label):
        """
        Gets mean and std for a speficic column

        Arguments:
            df (Pandas DataFrame): dataframe with the column label
            column_label (str): label of column which contains the value being tested

        Returns:
            stats (dict): dict of mean and standard deviation of column
        """
        mean = df[column_label].mean()
        std = df[column_label].std()
        stats = {"mean": mean, "std": std}
        return stats

    def _calc_t_test(self, column_label):
        """
        Performs a ttest comparison for a specific numerical column

        Arguments:
            column_label (str): label of column which contains the value being tested

        Returns:
            t_test_data (list): list with the value of the column, mean and std for both finance and non-finance,
                the t-test statistic, and its p-value
        """
        sample_stats = self._calc_t_test_stats(self.patients_w_tags, column_label)
        population_stats = self._calc_t_test_stats(self.patients_wout_tags, column_label)

        t_test = ttest_1samp(self.patients_w_tags[column_label].values, population_stats["mean"], nan_policy='omit')
        t_test_data = [column_label,
                       '{0:.2f} ({1:.1f})'.format(sample_stats["mean"], sample_stats["std"]),
                       '{0:.2f} ({1:.1f})'.format(population_stats["mean"], population_stats["std"]),
                       '{0:.3f}'.format(t_test.statistic),
                       self._format_p_value(t_test.pvalue)]
        return t_test_data

    def create_numerical_table(self):
        """
        Prints a summary table for t-test comparisons of numerical values at the patient level
        """
        numerical_table_data = [self._calc_t_test(column_label) for column_label in self._numerical_columns]
        print(tabulate(numerical_table_data, headers=['Feature', 'Mean (SD)', 'Mean (SD)', "Student's T", 'P-Value']))

    def create_all_tables(self):
        """
        Runs all table creation functions in class
        """
        self.create_population_stats_table()
        print('\n')
        self.create_summary_table()
        print('\n')
        self.create_numerical_table()
        print('\n')
        self.create_categorical_table()
        print('\n')
        self.create_logistic_regression_table()

    def create_logistic_regression_table(self, null_columns=['zip_median_income']):
        """
        Runs a logistic regression on selected categorical and numerical features and prints out a formatted table
        """

        self.regression_data = self.notes_data.sort_values(
            self._prediction_column, ascending=False).drop_duplicates(self._patient_id_column)
        self.regression_data = self.regression_data.merge(
            self._note_counts, how='inner', on=self._patient_id_column)

        for col in null_columns:
            self.regression_data = self.regression_data[self.regression_data[col].notnull()]

        # creat matrix of training features
        self.training_features = pd.concat([pd.get_dummies(self.regression_data[col], prefix=col)
                                           for col in self._categorical_columns] +
                                           [self.regression_data[col]
                                            for col in self._numerical_columns + ['note_count']],
                                           axis=1)

        # drop columns to allow for regression convergence
        self.training_features.drop(self._features_to_exclude, axis=1, inplace=True)
        self.training_features['intercept'] = 1.0

        # fit model
        logit = sm.Logit(self.regression_data[self._prediction_column], self.training_features)
        self.result = logit.fit()

        # create dataframe of results
        regression_results = pd.DataFrame()
        regression_results['odds_ratio'] = np.exp(self.result.params)
        regression_results['lower_bound'] = np.exp(self.result.conf_int()[0])
        regression_results['upper_bound'] = np.exp(self.result.conf_int()[1])
        regression_results['p_value'] = self.result.pvalues

        # format data for regression table
        regression_table_data = []
        for index, row in regression_results.iterrows():
            data_point = [index,
                          '{0:.2f} ({1:.2f}-{2:.2f})'.format(row['odds_ratio'], row['lower_bound'], row['upper_bound']),
                          self._format_p_value(row['p_value'])]
            regression_table_data.append(data_point)

        print(tabulate(regression_table_data, headers=['Feature', 'Odds Ratio (95% CI)', 'P Value']))
