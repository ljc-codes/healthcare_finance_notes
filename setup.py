import setuptools


setuptools.setup(
    name="notetagger",
    version="0.0.0",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'dill',
        'nltk',
        'tqdm',
        'sklearn',
        'prompt_toolkit',
        'pymongo',
    ],
    extras_require={
        'dev': [
            'psycopg2-binary',
            'ipython',
            'tensorflow',
            'keras',
        ],
        'partners': [
            'tabulate',
            'numpy',
            'scipy',
            'statsmodels',
        ],
    },
    entry_points={
        'console_scripts': {

            # extract data
            'extract-data=training.data_extraction:main',

            # train random forest
            'train-random-forest=training.random_forest.notetaggerrandomforest:train_random_forest',

            # create embedding layer
            'create-embedding=training.deep_learning.word_embeddings.create_embedding_matrix:main',

            # train lstm
            'train-lstm=training.deep_learning.lstm.notetaggerlstm:train_lstm',

            # run predictions and save to file
            'predict-tags=scripts.tag_notes:main',

            # validate data
            'note-tagger-viewer=notetagger.notetagger:main',

            # display training results
            'display-training-results=scripts.display_training_results:display_training_results',
        }
    }
)
