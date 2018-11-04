import setuptools
from setuptools.command.install import install


class DownloadStopWords(install):
    """
    install nltk stop words after installation
    """
    def run(self):
        import nltk
        print('test logging')
        nltk.download('stopwords')
        install.run(self)


setuptools.setup(
    name="notetagger",
    version="0.0.0",
    packages=setuptools.find_packages(),
    install_requires=[
        'pandas',
        'dill',
        'nltk',
        'tqdm',
    ],
    cmdclass={
        'install': DownloadStopWords
    },
    extras_require={
        'training': [
            'psycopg2',
            'sklearn',
            'pymongo'
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
