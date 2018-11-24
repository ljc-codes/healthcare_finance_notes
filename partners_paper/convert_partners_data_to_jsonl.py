import pandas as pd
from tqdm import tqdm


def convert_data(filepath):
    """
    Converts .txt note data to jsonl for use by the `notetagger` library

    Arguments:
        filepath (str): path to .txt notes file
    """
    all_data = []
    data_point = {}
    headers = ['subject_num', 'note_id', 'note_date', 'note_text']
    with open(filepath, 'r') as f:

        # loop through each line in text
        for i, line in tqdm(enumerate(f)):
            if i > 0:  # skip header line

                # if at the beginning of a new record, add the metadata columns to a dict
                if len(data_point) == 0:
                    for j, header in enumerate(headers[:3]):
                        data_point[header] = line.split('|')[j]

                    # if there is any note text in the line, initialize the key with it, otherwise initialize with blank
                    if len(line.split('|')) > 3:
                        data_point['note_text'] = line.split('|')[3]
                    else:
                        data_point['note_text'] = ''

                # it the '[report_end]' tag exists in the line, start up a new data point
                elif '[report_end]' in line:
                    data_point['note_text'] += line
                    all_data.append(data_point)
                    data_point = {}
                    continue

                # append all line text to the note text
                else:
                    data_point['note_text'] += line

    # initialize dataframe with data and save it to jsonl
    df = pd.DataFrame(all_data)
    df.to_json(filepath.replace('.txt', '.jsonl'), orient='records', lines=True)


if __name__ == '__main__':
    """
    Command line handler for `convert_data` function
    """
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--filepath',
                        '-f',
                        required=True,
                        type=str,
                        help='Path to data file')

    args = parser.parse_args()
    convert_data(args.filepath)
