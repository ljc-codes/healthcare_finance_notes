import dill as pickle

from utils import text_processing
from utils import constants


class NoteTagger:

    def __init__(self,
                 data,
                 text_column_name,
                 metadata_columns,
                 window_size=100,
                 tags=constants.TAGS,
                 stride_length=50,
                 feature_column_name=constants.FEATURE_COLUMN_NAME,
                 prediction_column_name=constants.PREDICTION_COLUMN_NAME):

        self._raw_data = data
        self._text_column_name = text_column_name
        self._metadata_columns = metadata_columns
        self._tags = tags
        self._window_size = window_size
        self._stride_length = stride_length
        self._feature_column_name = feature_column_name
        self._prediction_column_name = prediction_column_name

    def process_text(self):

        self._clean_data = text_processing.process_text(df=self._raw_data,
                                                        window_size=self._window_size,
                                                        tags=self._tags,
                                                        stride_length=self._stride_length,
                                                        text_column_name=self._text_column_name,
                                                        columns_to_keep=self._metadata_columns,
                                                        feature_column_name=self._feature_column_name)

    def transform_data(self):

        print("Transforming dataset with tfidf...")
        X_full = self._vectorizer.transform(self._clean_data[self._feature_column_namefeature_column_name])

        print("Reducing dimensionality with pca...")
        self._X = self._pca.transform(X_full)

    def predict_flag(self):
        self._clean_data[self._prediction_column_name] = self._model.predict(self._X)

    def aggregate_predictions(self):
        self._note_predictions = (self._clean_data[self._metadata_columns + [self._prediction_column_name]]
                                  .groupby(self._metadata_columns)[self._prediction_column_name].max())
