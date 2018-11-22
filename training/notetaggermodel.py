from datetime import datetime
import os

from pymongo import MongoClient
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

from notetagger import text_processing
from notetagger import constants


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
                 tokens_column_name=constants.FEATURE_COLUMN_NAME,
                 grid_search=False):

        self._model_id = model_name + datetime.today().strftime('%Y-%m-%dv%H-%m')
        self._raw_data = data
        self._text_column_name = text_column_name
        self._outcome_column_name = outcome_column_name
        self._model_save_file = os.path.join(model_save_path, self._model_id) + '.pkl'

        self._config = {"notetagger_params": {}}
        self._config["notetagger_params"]['window_size'] = window_size
        self._config["notetagger_params"]['word_tags'] = word_tags
        self._config["notetagger_params"]['stride_length'] = stride_length
        self._config["notetagger_params"]['grid_search'] = grid_search

    def _tokenize_text(self, raw_data):

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
        y = data[self._outcome_column_name].values
        return y

    def _grid_search_model(self, X_train, y_train):
        """
        Trains a model grid searching over provided parameters and saving the best ones to a config file
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

    def _fit_model(self,
                   X_train,
                   y_train):
        """
        Fits a model to training data
        """
        print("Training Model")
        if self._config["notetagger_params"]['grid_search']:
            self._grid_seach_model()

        self._model = self._base_model(**self._config['model_params']['model_config'])
        self._model.fit(X_train, y_train)

    def train_model(self, validation_data=None, store_result=True):
        X_train, y_train = self._process_text(raw_data=self._raw_data)
        self._fit_model(X_train=X_train, y_train=y_train)
        self._create_saved_model()
        if validation_data is not None:
            self._validate_model(validation_data=validation_data, store_result=store_result)

    def _validate_model(self, validation_data, store_result=True):
        print("Validating Model")
        note_tag_predictions = self._trained_model.predict_tag(
            data=validation_data,
            text_column_name=self._text_column_name,
            metadata_columns=[self._outcome_column_name],
            prediction_column_name='y_pred'
        )

        y_val = note_tag_predictions[self._outcome_column_name]
        y_pred_prob = note_tag_predictions['y_pred']

        self._model_validation_result = {
            "model_id": self._model_id,
            "config": self._config,
            "performance_metrics": {
                "auc": '{:.4f}'.format(roc_auc_score(y_true=y_val, y_score=y_pred_prob)),
                "metrics_by_threshold": []
            }
        }

        for threshold in range(3, 10):
            threshold /= 10
            y_pred = y_pred_prob > threshold
            self._model_validation_result["performance_metrics"]["metrics_by_threshold"].append(
                {"threshold": '{:.1f}'.format(threshold),
                 "accuracy": '{:.4f}'.format(accuracy_score(y_true=y_val, y_pred=y_pred)),
                 "precision": '{:.4f}'.format(precision_score(y_true=y_val, y_pred=y_pred)),
                 "recall": '{:.4f}'.format(recall_score(y_true=y_val, y_pred=y_pred))}
            )

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
        client = MongoClient(os.environ["MONGO_CONFIG"])
        db = client[db_name]
        collection = db[collection_name]
        result_id = collection.insert_one(self._model_validation_result).inserted_id
        print("Result {} saved".format(result_id))

    def _process_text(self, raw_data):
        raise NotImplementedError

    def _create_saved_model(self):
        raise NotImplementedError


class NoteTaggerTrainedModel:

    def __init__(self,
                 window_size,
                 model_config,
                 word_tags=constants.TAGS,
                 stride_length=None):

        self._window_size = window_size
        self._word_tags = word_tags
        self._stride_length = stride_length
        self._config = model_config

    def predict_tag(self, data, text_column_name, metadata_columns, prediction_column_name):
        raise NotImplementedError
