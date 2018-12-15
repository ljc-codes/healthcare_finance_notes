import json
import argparse

import dill as pickle
import pandas as pd
import numpy as np
from keras import layers, Model

from training.notetaggermodel import NoteTaggerModelTrain
from training.notetaggermodel import NoteTaggerTrainedModel
from training import text_processing
from notetagger import constants


class NoteTaggerLSTMTrain(NoteTaggerModelTrain):

    def __init__(self,
                 lstm_config_path,
                 data,
                 text_column_name,
                 outcome_column_name,
                 window_size,
                 model_save_path,
                 model_name='lstm',
                 word_tags=constants.TAGS,
                 stride_length=None,
                 grid_search=False):
        """
        Implements the NoteTaggerModelTrain class for a Random Forest Model. Most Arguments and
        Keyword Arguments inherited from the parent class

        Arguments:
            lstm_config_path (str): path to json file with random forest configuration parameters
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
        with open(lstm_config_path, 'r') as f:
            self._config["model_params"] = json.load(f)

        with open(self._config['model_params']['embedding_path'], 'rb') as embedding_file:
            self._embedding_layer = pickle.load(embedding_file)

        with open(self._config['model_params']['word_to_index'], 'r') as word_to_index_file:
            self._word_to_index = json.load(word_to_index_file)

        # set base model to random forest
        self._create_model()

    def _create_model(self):
        input_layer = layers.Input(shape=(self._config["notetagger_params"]['window_size'] * 2,), name='input_layer')
        model_layer = self._embedding_layer(input_layer)
        for i, lstm_layer in enumerate(self._config['model_params']['lstm_layers']):
            return_sequences = True if i < len(self._config['model_params']['lstm_layers']) - 1 else False
            model_layer = layers.Bidirectional(
                layers.LSTM(
                    lstm_layer,
                    return_sequences=return_sequences,
                    name='lstm_layer_{}'.format(i)))(model_layer)
        dense_layer = layers.Dense(1, name='dense_layer')(model_layer)
        output_layer = layers.Activation('sigmoid', name='activation_layer')(dense_layer)
        self._model = Model(input_layer, output_layer)
        self._model.compile(**self._config['model_params']['compile_config'])
        print(self._model.summary())

    def _token_to_index(self, tokenized_data):
        unk_token = self._word_to_index['unk']
        indexed_data = tokenized_data['tokenized_text'].map(lambda tokens: [self._word_to_index.get(token, unk_token)
                                                                            for token in tokens])
        max_size = self._config["notetagger_params"]['window_size'] * 2
        padded_data = indexed_data.map(lambda tokens: tokens + [0] * (max_size - len(tokens)))
        X = np.array(padded_data.tolist())
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
        X_train = self._token_to_index(tokenized_data=tokenized_data)
        y_train = self._get_outcome_value(data=tokenized_data)
        return X_train, y_train

    def _create_saved_model(self):
        """
        Creates and saves a `NoteTaggerTrainedRandomForest` class object with the necessary
        components
        """
        print("Saving Model")

        # initialize trained random forest class
        self._trained_model = NoteTaggerTrainedLSTM(
            window_size=self._config["notetagger_params"]['window_size'],
            word_tags=self._config["notetagger_params"]['word_tags'],
            stride_length=self._config["notetagger_params"]['stride_length'],
            model_config=self._config['model_params'])

        # set word_to_index
        self._trained_model._word_to_index = self._word_to_index
        self._trained_model._model = self._model

        # save model to pickle file
        with open(self._model_save_file, 'wb') as outfile:
            pickle.dump(self._trained_model, outfile)


def train_lstm():
    """
    Command line handler for NoteTaggerRandomForestTrain
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name',
                        '-mn',
                        default='lstm',
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

    parser.add_argument('--lstm_config',
                        '-lstm',
                        required=True,
                        type=str,
                        help='path to json file with lstm configuration parameters')

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
    lstm_trainer = NoteTaggerLSTMTrain(
        lstm_config_path=args.lstm_config,
        data=train_data,
        text_column_name=args.text_column_name,
        outcome_column_name=args.outcome_column_name,
        window_size=args.window_size,
        model_save_path=args.model_save_path,
        model_name=args.model_name,
        word_tags=args.word_tags,
        stride_length=args.stride_length,
        grid_search=False)

    # load validation data and train model
    validation_data = pd.read_json(args.validation_data_path, orient='records', lines=True)
    lstm_trainer.train_model(validation_data=validation_data)


class NoteTaggerTrainedLSTM(NoteTaggerTrainedModel):

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

        # index data
        unk_token = self._word_to_index['unk']
        indexed_data = tokenized_data['tokenized_text'].map(lambda tokens: [self._word_to_index.get(token, unk_token)
                                                                            for token in tokens])
        # pad data
        max_size = self._window_size * 2
        padded_data = indexed_data.map(lambda tokens: tokens + [0] * (max_size - len(tokens)))

        # create X
        X = np.array(padded_data.tolist())

        # predict text probability
        tokenized_data[prediction_column_name] = self._model.predict(X)

        # add '_id' column created during tokenization with metadata columns
        metadata_columns = ["_id"] + metadata_columns

        # aggregate predictions to each record
        note_tag_predictions = (tokenized_data[metadata_columns + [prediction_column_name]]
                                .groupby(metadata_columns)[prediction_column_name].max()
                                .reset_index())

        return note_tag_predictions
