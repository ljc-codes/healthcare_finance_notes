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
    ],
    extras_require={
        'training': [
            'psycopg2',
            'pymongo',
            'ipython',
        ]
    },
    entry_points={
        'console_scripts': {

            # extract data
            'extract-data=training.data_extraction:main',

            # train random forest
            'train-rf=training.random_forest.rf_model_training:main'

            # validate model
            'validate-model=training.validate_model:main',

            # display training results
            'display-training-results=scripts.display_training_results:display_training_results',
        }
    }
)
