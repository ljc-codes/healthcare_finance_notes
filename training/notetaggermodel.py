from sklearn.model_selection import GridSearchCV

from notetagger import text_processing
from notetagger import constants


class NoteTaggerModelTrain:

    def __init__(self,
                 data,
                 text_column_name,
                 outcome_column_name,
                 window_size,
                 model_save_path,
                 word_tags=constants.TAGS,
                 stride_length=None,
                 tokens_column_name=constants.FEATURE_COLUMN_NAME,
                 grid_search=False):
        super().__init__()
        self._raw_data = data
        self._text_column_name = text_column_name
        self._outcome_column_name = outcome_column_name
        self._window_size = window_size
        self._model_save_path = model_save_path
        self._word_tags = word_tags
        self._stride_length = stride_length
        self._grid_search = grid_search

    def _tokenize_text(self):

        self._tokenized_data = text_processing.process_text(df=self._raw_data,
                                                            window_size=self._window_size,
                                                            tags=self._word_tags,
                                                            stride_length=self._stride_length,
                                                            text_column_name=self._text_column_name,
                                                            columns_to_keep=self._outcome_column_name,
                                                            feature_column_name='tokenized_text')

        self.y_train = self._tokenized_data[self._outcome_column_name].values

    def _grid_search_model(self):
        """
        Trains a model grid searching over provided parameters and saving the best ones to a config file
        """

        # initialize grid search object
        self._model_cv = GridSearchCV(estimator=self._base_model,
                                      param_grid=self._config['grid_search']['parameters'],
                                      scoring=self._config['grid_search']['scoring_metric'],
                                      cv=self._config['grid_search']['cv_folds'],
                                      verbose=5)
        self._model_cv.fit(self._X_train, self._y_train)

        # select best performing model
        self._config['model_config'] = self._model_cv.best_params_

    def _fit_model(self,
                   data_input):
        """
        Fits a model to training data
        """
        if self._grid_search:
            self._grid_seach_model()

        self._model = self._base_model(**self._config['model_config'])
        self._model.fit(self._X_train, self._y_train)


class NoteTaggerTrainedModel:

    def __init__(self,
                 window_size,
                 config,
                 word_tags=constants.TAGS,
                 stride_length=None):
        self._window_size = window_size
        self._word_tags = word_tags
        self._stride_length = stride_length
        self._config = config

    def predict_tag(self, data, text_column_name, metadata_columns, prediction_column_name):
        raise NotImplementedError
