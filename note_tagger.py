import os
import dill as pickle

from utils import text_processing
from utils import constants


class NoteTagger:

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

        self._load_model_and_components()
        self.tag_notes()

    def _load_model_and_components(self):
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

        self._clean_data = text_processing.process_text(df=self._raw_data,
                                                        window_size=self.text_window_size,
                                                        tags=self._word_tags,
                                                        stride_length=self._stride_length,
                                                        text_column_name=self._text_column_name,
                                                        columns_to_keep=self._metadata_columns,
                                                        feature_column_name=self._feature_column_name)

    def _transform_data(self):

        if self._verbose:
            print("Transforming dataset with tfidf...")
        X_full = self._vectorizer.transform(self._clean_data[self._feature_column_name])

        if self._verbose:
            print("Reducing dimensionality with pca...")
        self._X = self._pca.transform(X_full)

    def _predict_tags(self):
        self._clean_data[self._prediction_column_name] = self._model.predict_proba(self._X)[:, 1]

    def _aggregate_predictions(self):
        self.note_tag_predictions = (self._clean_data[self._metadata_columns + [self._prediction_column_name]]
                                     .groupby(self._metadata_columns)[self._prediction_column_name].max()
                                     .reset_index())
        self.note_tag_predictions_all = self._merge_original_dataset()

    def tag_notes(self):
        self._process_text()
        self._transform_data()
        self._predict_tags()
        self._aggregate_predictions()

    def set_model_window_size(self, window_size):
        self._model_window_size = window_size
        self._load_model_and_components()

    def _merge_original_dataset(self, columns_to_include=[]):
        merged_data = self.note_tag_predictions.merge(
            self._raw_data[self._metadata_columns + columns_to_include],
            how='right',
            on=self._metadata_columns)

        return merged_data

    def save_predictions(self, save_filepath, with_full_text=False):
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
        coverage = len(self.note_tag_predictions.index) / len(self._raw_data.index) * 100
        print("Coverage: {0:.2f}%".format(coverage))

        positive_tags = (len(self.note_tag_predictions
                             [self.note_tag_predictions[self._prediction_column_name] > threshold].index) /
                         len(self.note_tag_predictions.index)) * 100
        print("Positive Tags: {0:.2f}%".format(positive_tags))
