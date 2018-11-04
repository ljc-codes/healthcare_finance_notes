import os
import dill as pickle

import text_processing
import constants


class NoteTagger:
    """
    This class tags notes as either having financial conversations or not using models trained on the MIMIC III dataset
    To use, first load a Pandas dataframe that has at least an id column and raw note text column,
    then initialize the class as follows:

    # initialize class
    tagger = NoteTagger(data=PandasDataframe,
                        text_column_name=str,
                        metadata_columns=[id_column_name,...])

    # print aggregate metrics
    tagger.quick_stats()

    # save predictions
    tagger.save_predictions(save_filepath=str)
    """

    def __init__(self,
                 data,
                 text_column_name,
                 metadata_columns,
                 model_window_size=100,
                 text_window_size=100,
                 word_tags=constants.TAGS,
                 stride_length=50,
                 feature_column_name=constants.FEATURE_COLUMN_NAME,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME,
                 verbose=False):
        """
        Intialize NoteTagger Object

        Arguments:
            data (Pandas DataFrame): pandas dataframe with at least an id column and text column
            text_column_name (str): label of column that contains raw note text
            metadata_columns (list of str): any column names to include with note tag predictions. Must be a list

        Keyword Arguments:
            model_window_size (int): size of text window that model was trained on, must be either 100, 200 or 500
            text_window_size (int): window size of text to tag, should ideally be the same as the `model_window_size`
                but can be different
            word_tags (list of str): List of specific terms to create windows around and tag.
                If `None`, a sliding window will be used over the entire note text. If not `None`, must be a list
            stride_length (int): If not using `word_tags`, stride length of sliding window
            feature_column_name (str): label of column that contains the created feature
            prediction_column_name (str): label of column that contains the predicted tag
            verbose (bool): print logging statements
        """

        # load inputs as class variables
        self._raw_data = data
        self._text_column_name = text_column_name
        self._metadata_columns = metadata_columns
        self._word_tags = word_tags
        self._model_window_size = model_window_size
        self.text_window_size = text_window_size
        self._stride_length = stride_length
        self._feature_column_name = feature_column_name
        self._prediction_column_name = prediction_column_name
        self._verbose = verbose

        # load models and tag notes
        self._load_model_and_components()
        self.tag_notes()

    def _load_model_and_components(self):
        """
        Loads the random forest model along with the tfidf vectorizer and pca
        """
        base_path = os.path.join(os.getcwd(), 'training/random_forest')

        # load model
        model_path = os.path.join(base_path, 'models/rf_ws{0}.pkl'.format(self._model_window_size))
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)

        # load vectorizer
        vectorizer_path = os.path.join(base_path, 'vectorizers/vectorizer_ws{0}.pkl'.format(self._model_window_size))
        with open(vectorizer_path, 'rb') as f:
            self._vectorizer = pickle.load(f)

        # load pca
        pca_path = os.path.join(base_path, 'pca/pca_ws{0}.pkl'.format(self._model_window_size))
        with open(pca_path, 'rb') as f:
            self._pca = pickle.load(f)

    def _process_text(self):
        """
        Tokenize and create windows around raw emr note text, store it in a new dataframe with
        columns for `feature_column_name` and `metadata_columns`
        """
        self._clean_data = text_processing.process_text(df=self._raw_data,
                                                        window_size=self.text_window_size,
                                                        tags=self._word_tags,
                                                        stride_length=self._stride_length,
                                                        text_column_name=self._text_column_name,
                                                        columns_to_keep=self._metadata_columns,
                                                        feature_column_name=self._feature_column_name)

    def _transform_data(self):
        """
        Takes tokenized data and transforms it into a tfidf matrix before reducing dimensionality with pca
        """

        if self._verbose:
            print("Transforming dataset with tfidf...")
        X_full = self._vectorizer.transform(self._clean_data[self._feature_column_name])

        if self._verbose:
            print("Reducing dimensionality with pca...")
        self._X = self._pca.transform(X_full)

    def _predict_tags(self):
        """
        Predicts probability of a note being a specific tag, assumes binary classification
        """
        self._clean_data[self._prediction_column_name] = self._model.predict_proba(self._X)[:, 1]

    def _aggregate_predictions(self):
        """
        Since the window approach generally yields multiple records per note, only the max prediction is taken
        over the entire note to determine if if corresponds to the specific tag. The predictions are then
        merged with the original dataset in case some predictions were unable to be made (because there were
        no `word_tags`) and the predictions for these records are marked as null
        """
        self.note_tag_predictions = (self._clean_data[self._metadata_columns + [self._prediction_column_name]]
                                     .groupby(self._metadata_columns)[self._prediction_column_name].max()
                                     .reset_index())
        self.note_tag_predictions_all = self._merge_original_dataset()

    def tag_notes(self):
        """
        Run the entire note tagging pipeline
        """
        self._process_text()
        self._transform_data()
        self._predict_tags()
        self._aggregate_predictions()

    def set_model_window_size(self, window_size):
        """
        Change the `model_window_size` and load corresponding models

        Arguments:
            window_size (int): size of window model was trained on
        """
        self._model_window_size = window_size
        self._load_model_and_components()

    def _merge_original_dataset(self, columns_to_include=[]):
        """
        Merge back in original dataset before tagging. Merge is done on `metadata_columns`

        Arguments:
            columns_to_include (list of str): columns to include in merge

        Returns:
            merged_data (Pandas DataFrame): pandas dataframe of merged notes
        """
        merged_data = self.note_tag_predictions.merge(
            self._raw_data[self._metadata_columns + columns_to_include],
            how='right',
            on=self._metadata_columns)

        return merged_data

    def save_predictions(self, save_filepath, with_full_text=False):
        """
        Save the predictions to a jsonl file

        Arguments:
            save_filepath (str): path to save dataset to

        Keyword Arguments:
            with_full_text (bool): include the original note full text in the saved down dataset
        """
        if with_full_text:
            predictions_w_text = self._merge_original_dataset(columns_to_include=[self._text_column_name])
            predictions_w_text.to_json(save_filepath,
                                       orient='records',
                                       lines=True)
        else:
            self.note_tag_predictions.to_json(save_filepath,
                                              orient='records',
                                              lines=True)

    def quick_stats(self, threshold=0.5):
        """
        Display the percentage of notes that were tagged (`coverage`, may be less than 100% if `word_tags` were used)
        as well as the percentage of notes that were positively tagged

        Keyword Arguments:
            threshold (float): probability threshold to be considered a positive tag
        """
        coverage = len(self.note_tag_predictions.index) / len(self._raw_data.index) * 100
        print("Coverage: {0:.2f}%".format(coverage))

        positive_tags = (len(self.note_tag_predictions
                             [self.note_tag_predictions[self._prediction_column_name] > threshold].index) /
                         len(self.note_tag_predictions.index)) * 100
        print("Positive Tags: {0:.2f}%".format(positive_tags))
