import pandas as pd

from scrap.GlobalVars import TRAIN_CSV_FILE, LF_CSV_FILE, TEST_CSV_FILE


def prepare_train_csv(TRAIN_CSV_FILE):
    pass


def prepare_test_csv(TEST_CSV_FILE):
    pass


def prepare_label_csv(LF_CSV_FILE):
    pass


class PrepareData:
    mainfile = ""

    def __init__(self, mainfile):
        self.mainfile = mainfile

    def prepare_data(self):
        main_file_df = pd.read_csv(self.mainfile)
        main_file_df['resolution'].apply(x lambda x: )
        prepare_train_csv(TRAIN_CSV_FILE)
        prepare_test_csv(TEST_CSV_FILE)
        prepare_label_csv(LF_CSV_FILE)