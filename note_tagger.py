import os
import dill as pickle

from utils import text_processing
from utils import constants


class NoteTagger:

    def __init__(self,
                 data,
                 text_column_name,
                 metadata_columns,
                 window_size=100,
                 word_tags=constants.TAGS,
                 stride_length=50,
                 feature_column_name=constants.FEATURE_COLUMN_NAME,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME):

        self._raw_data = data
        self._text_column_name = text_column_name
        self._metadata_columns = metadata_columns
        self._word_tags = word_tags
        self._window_size = window_size
        self._stride_length = stride_length
        self._feature_column_name = feature_column_name
        self._prediction_column_name = prediction_column_name

    def _load_model_and_components(self):
        base_path = os.path.join(os.cwd(), 'random_forest')

        # load model
        model_path = os.path.join(base_path, 'models/rf_ws{0}.pkl'.format(self._window_size))
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)

        # load vectorizer
        vectorizer_path = os.path.join(base_path, 'vectorizers/vectorizer_ws{0}.pkl'.format(self._window_size))
        with open(vectorizer_path, 'rb') as f:
            self._vectorizer = pickle.load(f)

        # load pca
        pca_path = os.path.join(base_path, 'pca/pca_ws{0}.pkl'.format(self._window_size))
        with open(pca_path, 'rb') as f:
            self._pca = pickle.load(f)

    def process_text(self):

        self.clean_data = text_processing.process_text(df=self._raw_data,
                                                       window_size=self._window_size,
                                                       tags=self._word_tags,
                                                       stride_length=self._stride_length,
                                                       text_column_name=self._text_column_name,
                                                       columns_to_keep=self._metadata_columns,
                                                       feature_column_name=self._feature_column_name)

    def transform_data(self):

        print("Transforming dataset with tfidf...")
        X_full = self._vectorizer.transform(self.clean_data[self._feature_column_namefeature_column_name])

        print("Reducing dimensionality with pca...")
        self.X = self._pca.transform(X_full)

    def predict_tags(self):
        self.clean_data[self._prediction_column_name] = self._model.predict(self.X)

    def aggregate_predictions(self):
        self.note_predictions = (self.clean_data[self._metadata_columns + [self._prediction_column_name]]
                                 .groupby(self._metadata_columns)[self._prediction_column_name].max())

    def get_tags(self):
        self._load_model_and_components()
        self.process_text()
        self.transform_data()
        self.predict_tags()
        self.aggregate_predictions()
        return self._note_predictions

    def set_window(self, window_size):
        self._window_size = window_size
        self._load_model_and_components()

    def merge_full_text(self):
        self.note_predictions_w_text = self.note_predictions.merge(
            self.raw_data[self._metadata_columns + [self._text_column_name]],
            how='inner',
            on=self._metadata_columns)

    def save_predictions(self, save_filepath, with_full_text=False):
        if with_full_text:
            self.merge_full_text()
            self.note_predictions_w_text.to_json(save_filepath,
                                                 orient='records',
                                                 lines=True)
        else:
            self.note_predictions.to_json(save_filepath,
                                          orient='records',
                                          lines=True)
