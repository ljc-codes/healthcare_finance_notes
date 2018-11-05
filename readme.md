# Financial Note Tagger
A class for tagging electronic medical record (EMR) notes as either having financial conversations or not. The tagger takes a `Pandas DataFrame` with at least note text and an id and returns a tag for each note. The model comes out of the work done in [**Prevalence and Nature of Financial Considerations Documented in Narrative Clinical Records in Intensive Care Units**](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2712180?resultClick=3)

## Setup
To use the notetagger ensure that you have `python 3.6` installed, clone this repo, and then run `pip install -e .`

If you'd also like to train a new model, also run `pip install -e .[training]`

## Usage
When tagging a new dataset, first load a [Pandas Dataframe](https://pandas.pydata.org/pandas-docs/stable/api.html#input-output) that has at least an id column and raw note text column, then initialize the class as follows:

```python
# initialize class
tagger = NoteTagger(data=PandasDataframe,
                    text_column_name=str,
                    metadata_columns=[id_column_name,...])

# save predictions
tagger.save_predictions(save_filepath=str)
```

If you'd like to tag notes that don't contain any of the [keywords](https://github.com/pateli18/healthcare_finance_notes/blob/master/notetagger/constants.py) (`cost`, `insurance`, `pay`, etc.) the model was trained on, initialize the class as follows:

```python
# initialize class
tagger = NoteTagger(data=PandasDataframe,
                    text_column_name=str,
                    metadata_columns=[id_column_name,...],
                    word_tags=None)
```