# Financial Note Tagger
A class for tagging electronic medical record (EMR) notes as either having financial conversations or not. The tagger takes a `Pandas DataFrame` with at least note text and an id and returns a tag for each note. The model comes out of the work done in [**Prevalence and Nature of Financial Considerations Documented in Narrative Clinical Records in Intensive Care Units**](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2712180?resultClick=3)

## Setup
To use the notetagger ensure that you have `python 3.6` installed, clone this repo, and then run `pip install -e .`

If you'd also like to train a new model, also run `pip install -e .[training]`

## Usage

### Tagging Notes

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

### Validating Tags

To ensure the model is making predictions as anticipated on an untagged dataset, use the command line interface provided by the `noteviewer.py` file. This will print out snippets of text surrounding the `word_tags` used to trian the model (if no word tags were used, the entire note is printed) and give the user an option to tag the note positively or negatively, saving the predictions to disk.

To launch the validator from the command line, type in the following command:

```bash
note-viewer -p 'path to predictions file' -o 'path to original dataset' -j note_id -t text
```

To launch the validator via python, you can do the following:

```python
# initialize class
tagger = NoteViewer(predictions_data=PandasDataframe,
                    original_data=PandasDataframe,
                    join_column_names=[str,...],
                    text_column_name=str)

# start validator
tagger.validate_predictions(validation_save_path=str)
```
