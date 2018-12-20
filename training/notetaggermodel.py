from datetime import datetime
import os

from pymongo import MongoClient
from sklearn.model_selection import GridSearchCV

from training import text_processing
from notetagger import constants
from notetagger import metrics_calculation


class NoteTaggerModelTrain:

    def __init__(self,
                 model_name,
                 data,
                 text_column_name,
                 outcome_column_name,
                 window_size,
                 model_save_path,
                 word_tags=constants.TAGS,
                 stride_length=None,
                 grid_search=False):
        """
        Intialize the NoteTaggerModelTrain class, which trains a model for note tagging. This class
        will need to be subclassed and the `_process_text` and `_create_saved_model` methods implemented.

        Arguments:
            model_name (str): name of model to use in model id calculation
            data (Pandas DataFrame): training data for model
            text_column_name (str): label for column with raw text
            outcome_column_name (str): label for outcome column
            window_size (int): size of window taken before and after text (total size of window * 2)
            model_save_path (str): folder to save model in

        Keyword Arguments:
            word_tags (list of str): list of word tags to search for in text and create windows around,
                if None a sliding window is used
            stride_length (int): stride for sliding window, used only if `word_tags` is `None`
            grid_search (bool): grid search over parameters before training the final model
        """
        self._model_name = model_name

        self._raw_data = data
        self._text_column_name = text_column_name
        self._outcome_column_name = outcome_column_name

        # create full path to model by including a pkl file with the model id
        self._model_save_file = os.path.join(model_save_path, self._model_id) + '.pkl'

        # initalize config dictionary
        self._config = {"notetagger_params": {}}
        self._config["notetagger_params"]['window_size'] = window_size
        self._config["notetagger_params"]['word_tags'] = word_tags
        self._config["notetagger_params"]['stride_length'] = stride_length
        self._config["notetagger_params"]['grid_search'] = grid_search

    @property
    def _model_id(self):
        # create model id by appending the datetime the model was trained
        return self._model_name + datetime.today().strftime('%Y-%m-%dv%H-%m')

    def _tokenize_text(self, raw_data):
        """
        Take raw text and tokenize the data with a given window size or using a sliding window
        depending whether or not word_tags were provided

        Arguments:
            raw_data (Pandas DataFrame): data with a raw text column and outcome column

        Returns:
            tokenized_data (Pandas DataFrame): data with a tokenized text column, outcome column, and
                _id column
        """

        print("Tokenizing Data")
        tokenized_data = text_processing.process_text(
            df=raw_data,
            window_size=self._config["notetagger_params"]['window_size'],
            tags=self._config["notetagger_params"]['word_tags'],
            stride_length=self._config["notetagger_params"]['stride_length'],
            text_column_name=self._text_column_name,
            columns_to_keep=[self._outcome_column_name],
            feature_column_name='tokenized_text')

        return tokenized_data

    def _get_outcome_value(self, data):
        """
        Creates an array of the outcome variable given a dataframe

        Arguments:
            data (Pandas DataFrame): dataframe with the `_outcome_column_name` in it, usually the
                result of `tokenize_text`

        Results:
            y (array): numpy array of the outcome variable
        """
        y = data[self._outcome_column_name].values
        return y

    def _grid_search_model(self, X_train, y_train):
        """
        Trains a model grid searching over provided parameters and saving the best ones to a config file

        Arguments:
            X_train (array): Array with training features
            y_train (array): Array with training outcomes
        """

        # initialize grid search object
        print("Running Grid Search")
        self._model_cv = GridSearchCV(estimator=self._base_model,
                                      param_grid=self._config['grid_search']['parameters'],
                                      scoring=self._config['grid_search']['scoring_metric'],
                                      cv=self._config['grid_search']['cv_folds'],
                                      verbose=5)
        self._model_cv.fit(X_train, y_train)

        # select best performing model
        self._config['model_params']['model_config'] = self._model_cv.best_params_
        self._model = self._base_model(**self._config['model_params']['model_config'])

    def _fit_model(self,
                   X_train,
                   y_train,
                   **kwargs):
        """
        Fits a model to training data

        Arguments:
            X_train (array): Array with training features
            y_train (array): Array with training outcomes
        """
        print("Training Model")
        if self._config["notetagger_params"]['grid_search']:
            self._grid_seach_model()

        self._model.fit(X_train, y_train, **kwargs)

    def train_model(self, validation_data=None, store_result=True, **kwargs):
        """
        Processes the raw data, trains a model, and validates it if validation data provided

        Keyword Arguments:
            validation_data (Pandas DataFrame): dataframe with at least the same columns as `_raw_data`.
                If `None` no validation is performed
            store_result (bool): store the validation result in a Mongo database
        """
        X_train, y_train = self._process_text(raw_data=self._raw_data)
        self._fit_model(X_train=X_train, y_train=y_train, **kwargs)
        self._create_saved_model()
        if validation_data is not None:
            self._validate_model(validation_data=validation_data, store_result=store_result)

    def _validate_model(self, validation_data, store_result=True):
        """
        Gets model tags on a hold out set and calculates various metrics associated with it

        Arguments:
            validation_data (Pandas DataFrame): dataframe with at least the same columns as `_raw_data`

        Keyword Arguments:
            store_result (bool): store the validation result in a Mongo database
        """
        print("Validating Model")

        # get model predictions
        note_tag_predictions = self._trained_model.predict_tag(
            data=validation_data,
            text_column_name=self._text_column_name,
            metadata_columns=[self._outcome_column_name],
            prediction_column_name='y_pred'
        )

        # extract predictions and actuals arrays
        y_val = note_tag_predictions[self._outcome_column_name]
        y_pred_prob = note_tag_predictions['y_pred']

        # create dictionary for storing model results
        self._model_validation_result = {
            "model_id": self._model_id,
            "config": self._config,
            "performance_metrics": metrics_calculation.calculate_performance_metrics(y_val, y_pred_prob)
        }

        if store_result:
            self._store_validation_result()

    def _store_validation_result(self,
                                 db_name=constants.MONGO_DATABASE_NAME,
                                 collection_name=constants.MONGO_COLLECTION_NAME):
        """
        Store model in MongoDB and prints id

        Keyword Arguments:
            db_name (str): name of database in MongoDB
            collection_name (str): name of collection in database
        """

        # load mongo collection
        client = MongoClient(os.environ["MONGO_CONFIG"])
        db = client[db_name]
        collection = db[collection_name]

        # add to mongo collection
        result_id = collection.insert_one(self._model_validation_result).inserted_id
        print("Result {} saved".format(result_id))

    def _process_text(self, raw_data):
        """
        Takes in a dataframe with raw note text and returns a tuple of training features
        (`X_train`) and training outcomes (`y_train`)

        Arguments:
            raw_data (Pandas DataFrame): data with a raw text column and outcome column

        Returns:
            X_train (array): Array with training features
            y_train (array): Array with training outcomes
        """
        raise NotImplementedError

    def _create_saved_model(self):
        """
        Save a `NoteTaggerTrainedModel` after training that implements the `predict_tag` method
        """
        raise NotImplementedError


class NoteTaggerTrainedModel:

    def __init__(self,
                 window_size,
                 model_config,
                 word_tags=constants.TAGS,
                 stride_length=None):
        """
        Initializes the `NoteTaggerTrainedModel` class which is used by `notetagger.py` to tag any EMR note
        dataset

        Arguments:
            window_size (int): size of window taken before and after text (total size of window * 2)
            model_config (dict): dict storing any metadata associated with model training

        Keyword Arguments:
            word_tags (list of str): list of word tags to search for in text and create windows around,
                if None a sliding window is used
            stride_length (int): stride for sliding window, used only if `word_tags` is `None`
        """

        self._window_size = window_size
        self._word_tags = word_tags
        self._stride_length = stride_length
        self._config = model_config

    def predict_tag(self, data, text_column_name, metadata_columns, prediction_column_name):
        """
        Uses the given model to predict tags on a given dataset and returns a dataframe with the
        predictions and metadata

        Arguments:
            data (Pandas Dataframe): data with a raw text column
            text_column_name (str): label for column with raw text
            metadata_columns (list of str): list of column names to include in output dataframe
            prediction_column_name (str): name of column with predictions

        Returns:
            note_tag_predictions (Pandas DataFrame): dataframe with a column for predictions and
                potentially other columns
        """
        raise NotImplementedError
