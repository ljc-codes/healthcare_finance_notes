# Financial Note Tagger
A class for tagging electronic medical record (EMR) notes as either having financial conversations or not. The tagger takes a `Pandas DataFrame` with at least note text and an id and returns a tag for each note. The model comes out of the work done in [**Prevalence and Nature of Financial Considerations Documented in Narrative Clinical Records in Intensive Care Units**](https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2712180?resultClick=3)

## Setup
To use the notetagger ensure that you have `python 3.6` installed, clone this repo, and then run `pip install -e .`

## Usage

### Tagging Notes

#### Command Line

When tagging a new dataset, the easiest way to do so is to run the following command:

```bash
predict-tags --model_path 'path to model' --input_data 'path to existing jsonl file' --output_data 'path to new jsonl file' --text_column_name 'name of column with raw text' --metadata_columns 'name of metadata column to include'
```

Note that `input_data` and `output_data` should be in a the `jsonl` format and that multiple `metadata_columns` can be used by replicating the tag

#### Python

You can also load any pickled model object and call the `predict_tags` function on a dataframe

```python
import dill as pickle

# load model
with open(path_to_model, 'rb') as f:
    model = pickle.load(f)

# create predictions dataset
predictions_data = model.predict_tag(
    data=data_with_raw_note_text,
    text_column_name=label_of_column,
    metadata_columns=list_of_column_labels)
```

### Validating Model Predictions

To ensure the model is making predictions as anticipated on an untagged dataset, use the command line interface provided by the `notetagger.py` file. This will print out snippets of text surrounding the `word_tags` used to trian the model (if no word tags were used, the entire note is printed) and give the user an option to tag the note positively or negatively, saving the predictions to disk.

#### Command Line

To launch the validator from the command line, type in the following command:

```bash
note-tagger --predictions_file_path 'path to predictions file' --original_file_path 'path to original dataset' --text_column_name 'name of column with raw text' --metadata_columns 'name of metadata column to include'
```

Note that `input_data` and `output_data` should be in a the `jsonl` format and that multiple `metadata_columns` can be used by replicating the tag. The `--run_predictions` flag can be used to run predictions before launching the viewer (be sure to include the `--model_path` flag as well)

#### Python

To launch the validator via python, you can do the following:

```python
# initialize class
tagger = NoteTagger(predictions_data=PandasDataframe,
                    original_data=PandasDataframe,
                    text_column_name=label_of_column,
                    metadata_columns=list_of_column_labels)

# start validator
tagger.validate_predictions(validation_save_path=path_to_jsonl_file)
```
