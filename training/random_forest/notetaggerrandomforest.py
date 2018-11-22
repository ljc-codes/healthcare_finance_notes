import json

import dill as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier

from training.notetaggermodel import NoteTaggerModelTrain
from training.notetaggermodel import NoteTaggerTrainedModel
from notetagger import text_processing
from notetagger import constants


class NoteTaggerRandomForestTrain(NoteTaggerModelTrain):

    def __init__(self,
                 rf_config_path,
                 data,
                 text_column_name,
                 outcome_column_name,
                 window_size,
                 model_save_path,
                 model_name='random_forest',
                 word_tags=constants.TAGS,
                 stride_length=None,
                 tokens_column_name=constants.FEATURE_COLUMN_NAME,
                 grid_search=False):

        super().__init__(model_name=model_name,
                         data=data,
                         text_column_name=text_column_name,
                         outcome_column_name=outcome_column_name,
                         window_size=window_size,
                         model_save_path=model_save_path,
                         word_tags=word_tags,
                         stride_length=stride_length,
                         tokens_column_name=tokens_column_name,
                         grid_search=grid_search)

        # load configuration file
        with open(rf_config_path, 'r') as f:
            self._config["model_params"] = json.load(f)

        self._base_model = RandomForestClassifier

    def _fit_tfidf(self, tokenized_data):
        """
        Fits a tf-idf vectorizer on a provided dataframe

        Arguments:
            data_input (Pandas Dataframe): dataframe with tokenized words in the `tokens_column_name` column
        """
        print("Training Vectorizer")

        if "ngram_range" in self._config["model_params"]['tfidf_config']:
            self._config["model_params"]['tfidf_config']['ngram_range'] = tuple(
                self._config["model_params"]['tfidf_config']["ngram_range"])

        self._vectorizer = TfidfVectorizer(analyzer='word',
                                           tokenizer=lambda doc: doc,
                                           preprocessor=lambda doc: doc,
                                           token_pattern=None,
                                           **self._config["model_params"]['tfidf_config'])

        vectorized_data = self._vectorizer.fit_transform(tokenized_data['tokenized_text'])
        return vectorized_data

    def _fit_pca(self, vectorized_data):
        """
        Fits an PCA component to a provided array

        Arguments:
            data_input (numpy array): matrix transformed by tfidf
            config (dict): config params for a given pca

        Returns:
            pca (TruncatedSVD): pca object for transforming any tfidf matrix
        """

        print("Using PCA")

        self._pca = TruncatedSVD(**self._config["model_params"]['pca_config'])

        X = self._pca.fit_transform(vectorized_data)
        return X

    def _process_text(self, raw_data):
        tokenized_data = self._tokenize_text(raw_data=raw_data)
        vectorized_data = self._fit_tfidf(tokenized_data=tokenized_data)
        X_train = self._fit_pca(vectorized_data=vectorized_data)
        y_train = self._get_outcome_value(data=tokenized_data)
        return X_train, y_train

    def _create_saved_model(self):
        print("Saving Model")
        self._trained_model = NoteTaggerTrainedRandomForest(
            window_size=self._config["notetagger_params"]['window_size'],
            word_tags=self._config["notetagger_params"]['word_tags'],
            stride_length=self._config["notetagger_params"]['stride_length'],
            model_config=self._config['model_params'])

        self._trained_model._vectorizer = self._vectorizer
        self._trained_model._pca = self._pca
        self._trained_model._model = self._model

        with open(self._model_save_file, 'wb') as outfile:
            pickle.dump(self._trained_model, outfile)


class NoteTaggerTrainedRandomForest(NoteTaggerTrainedModel):

    def __init__(self,
                 window_size,
                 model_config,
                 word_tags=constants.TAGS,
                 stride_length=None):

        super().__init__(window_size=window_size,
                         model_config=model_config,
                         word_tags=word_tags,
                         stride_length=stride_length)

    def predict_tag(self, data, text_column_name, metadata_columns, prediction_column_name):
        tokenized_data = text_processing.process_text(df=data,
                                                      window_size=self._window_size,
                                                      tags=self._word_tags,
                                                      stride_length=self._stride_length,
                                                      text_column_name=text_column_name,
                                                      columns_to_keep=metadata_columns,
                                                      feature_column_name='tokenized_text')

        vectorized_data = self._vectorizer.transform(tokenized_data['tokenized_text'])
        X = self._pca.transform(vectorized_data)
        tokenized_data[prediction_column_name] = self._model.predict_proba(X)[:, 1]

        metadata_columns = ["_id"] + metadata_columns

        note_tag_predictions = (tokenized_data[metadata_columns + [prediction_column_name]]
                                .groupby(metadata_columns)[prediction_column_name].max()
                                .reset_index())

        return note_tag_predictions
