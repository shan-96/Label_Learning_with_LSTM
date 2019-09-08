import csv

import pandas as pd

from scrap.GlobalVars import TRAIN_CSV_FILE, LF_CSV_FILE, TEST_CSV_FILE


def prepare_train_csv(main_file_df):
    split = int(len(main_file_df) * 0.8)
    sample_df = main_file_df.iloc[0:split]
    sample_df['resolution'] = sample_df['resolution'].apply(lambda x: '')
    sample_df.to_csv(index=False, path_or_buf=TRAIN_CSV_FILE, quoting=csv.QUOTE_NONE)


def prepare_test_csv(main_file_df):
    split1 = int(len(main_file_df) * 0.8)
    split2 = int(len(main_file_df) * 0.9)
    sample_df = main_file_df.iloc[split1:split2]
    sample_df.to_csv(index=False, path_or_buf=TEST_CSV_FILE, quoting=csv.QUOTE_NONE)


def prepare_label_csv(main_file_df):
    split = int(len(main_file_df) * 0.1)
    sample_df = main_file_df.tail(split)
    sample_df.to_csv(index=False, path_or_buf=LF_CSV_FILE, quoting=csv.QUOTE_NONE)


class PrepareData:
    mainfile = ""

    def __init__(self, mainfile):
        self.mainfile = mainfile

    def prepare_data(self):
        main_file_df = pd.read_csv(self.mainfile)
        fixed_list = ['Fixed', 'Done', 'Fixed', 'Fixed in a Later Version', 'Enhancement implemented',
                      'Training Provided', 'Working as Expected', 'Request Completed', 'Not an Issue']
        main_file_df['resolution'] = main_file_df['resolution'].apply(lambda x: 1 if x in fixed_list else 0)

        # large number of unlabelled tickets
        prepare_train_csv(main_file_df)

        # small number of labelled tickets
        prepare_test_csv(main_file_df)

        # labelled tickets for building LF
        prepare_label_csv(main_file_df)
