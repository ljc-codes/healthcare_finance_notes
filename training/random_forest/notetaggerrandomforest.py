import json
import argparse

import dill as pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

from training.notetaggermodel import NoteTaggerModelTrain
from training.notetaggermodel import NoteTaggerTrainedModel
from training import text_processing
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
                 grid_search=False):
        """
        Implements the NoteTaggerModelTrain class for a Random Forest Model. Most Arguments and
        Keyword Arguments inherited from the parent class

        Arguments:
            rf_config_path (str): path to json file with random forest configuration parameters
        """

        super().__init__(model_name=model_name,
                         data=data,
                         text_column_name=text_column_name,
                         outcome_column_name=outcome_column_name,
                         window_size=window_size,
                         model_save_path=model_save_path,
                         word_tags=word_tags,
                         stride_length=stride_length,
                         grid_search=grid_search)

        # load configuration file
        with open(rf_config_path, 'r') as f:
            self._config["model_params"] = json.load(f)

        # set base model to random forest
        self._base_model = RandomForestClassifier

    def _fit_tfidf(self, tokenized_data):
        """
        Fits a tf-idf vectorizer on tokenized text data

        Arguments:
            tokenized_data (Pandas Dataframe): dataframe with tokenized words in the 'tokenized_text' column

        Returns:
            vectorized_data (array): tfidf matrix of tokenized words
        """
        print("Training Vectorizer")

        # convert ngram range to tuple
        if "ngram_range" in self._config["model_params"]['tfidf_config']:
            self._config["model_params"]['tfidf_config']['ngram_range'] = tuple(
                self._config["model_params"]['tfidf_config']["ngram_range"])

        # initialize vectorizer
        self._vectorizer = TfidfVectorizer(analyzer='word',
                                           tokenizer=lambda doc: doc,
                                           preprocessor=lambda doc: doc,
                                           token_pattern=None,
                                           **self._config["model_params"]['tfidf_config'])

        vectorized_data = self._vectorizer.fit_transform(tokenized_data['tokenized_text'])
        return vectorized_data

    def _fit_pca(self, vectorized_data):
        """
        Fits an PCA component to a tfidf vectorized data array

        Arguments:
            vectorized_data (array): tfidf matrix of tokenized words

        Returns:
            X (array): feature array to be used in training / validating model
        """

        print("Using PCA")

        # initalize truncated svd
        self._pca = TruncatedSVD(**self._config["model_params"]['pca_config'])

        X = self._pca.fit_transform(vectorized_data)
        return X

    def _process_text(self, raw_data):
        """
        Takes in a dataframe with raw note text and training features and outcomes by first
        tokenizing the text, then transforming it with tfidf before reducing dimensionality with
        pca

        Arguments:
            raw_data (Pandas DataFrame): data with a raw text column and outcome column

        Returns:
            X_train (array): Array with training features
            y_train (array): Array with training outcomes
        """
        tokenized_data = self._tokenize_text(raw_data=raw_data)
        vectorized_data = self._fit_tfidf(tokenized_data=tokenized_data)
        X_train = self._fit_pca(vectorized_data=vectorized_data)
        y_train = self._get_outcome_value(data=tokenized_data)
        return X_train, y_train

    def _create_saved_model(self):
        """
        Creates and saves a `NoteTaggerTrainedRandomForest` class object with the necessary
        components
        """
        print("Saving Model")

        # initialize trained random forest class
        self._trained_model = NoteTaggerTrainedRandomForest(
            window_size=self._config["notetagger_params"]['window_size'],
            word_tags=self._config["notetagger_params"]['word_tags'],
            stride_length=self._config["notetagger_params"]['stride_length'],
            model_config=self._config['model_params'])

        # set vectorizer, pca, and model variables to class
        self._trained_model._vectorizer = self._vectorizer
        self._trained_model._pca = self._pca
        self._trained_model._model = self._model

        # save model to pickle file
        with open(self._model_save_file, 'wb') as outfile:
            pickle.dump(self._trained_model, outfile)


def train_random_forest():
        parser = argparse.ArgumentParser()

        parser.add_argument('--model_name',
                            '-mn',
                            default='random_forest',
                            type=str,
                            help='name of model to use in model id calculation')

        parser.add_argument('--text_column_name',
                            '-t',
                            default=constants.TEXT_COLUMN_NAME,
                            type=str,
                            help='label for column with raw text')

        parser.add_argument('--outcome_column_name',
                            '-o',
                            default=constants.OUTCOME_COLUMN_NAME,
                            type=str,
                            help='label for outcome column')

        parser.add_argument('--window_size',
                            '-ws',
                            required=True,
                            type=int,
                            help='size of window taken before and after text (total size of window * 2)')

        parser.add_argument('--model_save_path',
                            '-mp',
                            required=True,
                            type=str,
                            help='folder to save model in')

        parser.add_argument('--word_tags',
                            '-wt',
                            default=constants.TAGS,
                            action='append',
                            type=str,
                            help='Tags on which the model predicted on')

        parser.add_argument('--stride_length',
                            '-s',
                            default=None,
                            type=int,
                            help='stride for sliding window, used only if `word_tags` is `None`')

        parser.add_argument('--grid_search',
                            '-gs',
                            default=False,
                            action='store_true',
                            help='grid search over parameters before training the final model')

        parser.add_argument('--random_forest_config',
                            '-rf',
                            required=True,
                            type=str,
                            help='path to json file with random forest configuration parameters')

        parser.add_argument('--training_data_path',
                            '-train',
                            required=True,
                            type=str,
                            help='path to training data, must be jsonl')

        parser.add_argument('--validation_data_path',
                            '-val',
                            required=True,
                            type=str,
                            help='path to validation data, must be jsonl')

        args = parser.parse_args()

        # load data and initialize model trainer
        train_data = pd.read_json(args.training_data_path, orient='records', lines=True)
        rf_trainer = NoteTaggerRandomForestTrain(
            rf_config_path=args.random_forest_config,
            data=train_data,
            text_column_name=args.text_column_name,
            outcome_column_name=args.outcome_column_name,
            window_size=args.window_size,
            model_save_path=args.model_save_path,
            model_name=args.model_name,
            word_tags=args.word_tags,
            stride_length=args.stride_length,
            grid_search=args.grid_search)

        # load validation data and train model
        validation_data = pd.read_json(args.validation_data_path, orient='records', lines=True)
        rf_trainer.train_model(validation_data=validation_data)


class NoteTaggerTrainedRandomForest(NoteTaggerTrainedModel):

    def __init__(self,
                 window_size,
                 model_config,
                 word_tags=constants.TAGS,
                 stride_length=None):
        """
        Implements the NoteTaggerTrainedModel class for a Random Forest Model. All Arguments and
        Keyword Arguments inherited from the parent class
        """

        super().__init__(window_size=window_size,
                         model_config=model_config,
                         word_tags=word_tags,
                         stride_length=stride_length)

    def predict_tag(self, data, text_column_name, metadata_columns, prediction_column_name):
        """
        Takes raw data and predicts tags for each record using the random forest model. Aggregates
        the multiple predictions per record (e.g. multiple windows per record) into one prediction
        and then returns the predictions with associated metadata in a dataframe

        Arguments:
            data (Pandas Dataframe): data with a raw text column
            text_column_name (str): label for column with raw text
            metadata_columns (list of str): list of column names to include in output dataframe
            prediction_column_name (str): name of column with predictions

        Returns:
            note_tag_predictions (Pandas Dataframe): dataframe with metadata_columns and prediction_column_name
        """

        # tokenize text, tfidf it, and pca it
        tokenized_data = text_processing.process_text(df=data,
                                                      window_size=self._window_size,
                                                      tags=self._word_tags,
                                                      stride_length=self._stride_length,
                                                      text_column_name=text_column_name,
                                                      columns_to_keep=metadata_columns,
                                                      feature_column_name='tokenized_text')

        vectorized_data = self._vectorizer.transform(tokenized_data['tokenized_text'])
        X = self._pca.transform(vectorized_data)

        # predict text probability
        tokenized_data[prediction_column_name] = self._model.predict_proba(X)[:, 1]

        # add '_id' column created during tokenization with metadata columns
        metadata_columns = ["_id"] + metadata_columns

        # aggregate predictions to each record
        note_tag_predictions = (tokenized_data[metadata_columns + [prediction_column_name]]
                                .groupby(metadata_columns)[prediction_column_name].max()
                                .reset_index())

        return note_tag_predictions
