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
            'psycopg2-binary',
            'pymongo',
            'ipython',
        ]
    },
    entry_points={
        'console_scripts': {

            # extract data
            'extract-data=training.data_extraction:main',

            # run predictions and save to file
            'predict-tags=scripts.tag_notes:main',

            # validate data
            'note-tagger-viewer=notetagger.notetagger:main',

            # display training results
            'display-training-results=scripts.display_training_results:display_training_results',
        }
    }
)
