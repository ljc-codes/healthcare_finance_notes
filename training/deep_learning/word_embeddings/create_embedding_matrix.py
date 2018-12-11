import os
import json
import re

import dill as pickle
from tqdm import tqdm
import numpy as np
from keras.layers import Embedding

from training.text_processing import STOPWORDS


def create_embedding_matrix(glove_filepath, save_base_path):
    regex = re.compile('[^a-zA-Z ]')

    word_to_index = {}
    embedding_list = []
    counter = 0
    with open(glove_filepath) as file:
        for word_data in tqdm(file):
            word_data_list = word_data.split()
            word = word_data_list[0]
            if word not in STOPWORDS and regex.sub('', word):
                word_to_index[word] = counter
                embedding_list.append([float(point) for point in word_data_list[1:]])
                counter += 1

    embedding_dims = glove_filepath.split('.')[-2]
    with open(os.path.join(save_base_path, 'word_to_index_{}.json'.format(embedding_dims)), 'w') as save_path:
        json.dump(word_to_index, save_path)

    embedding_matrix = np.asarray(embedding_list)
    layer = Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        trainable=False
    )
    with open(os.path.join(save_base_path, 'embedding_matrix_{}.pkl'.format(embedding_dims)), 'wb') as save_path:
        pickle.dump(layer, save_path)
