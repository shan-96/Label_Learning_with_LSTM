import re

import numpy as np
import pandas as pd
from displayfunction import display
from metal import MajorityLabelVoter
from metal.analysis import lf_summary, label_coverage
from metal.label_model import LabelModel
from scipy import sparse
from sklearn.metrics import classification_report
from snorkel.labeling import labeling_function


class CommentSnorkel:
    # voting scores
    POSITIVE = 1
    NEGATIVE = -1
    ABSTAIN = 0

    # labels/JIRA resolutions
    labels = {
        'Fixed': 0,
        'As Designed': 1,
        'Training': 2,
        'Not an Issue': 3,
        'Request Completed': 4,
        'Deferred': 5,
        'Duplicate': 6,
        'Cannot Reproduce': 7,
        '3rd Party Request Completed': 8,
        'Lack of Response': 9,
        'ASP Request Completed': 10,
        'Implementation': 11,
        'Design Gap': 12,
        'Fixed in a Later Version': 13,
        'Unresolved': 14,
        'Feature Request': 15,
        'Dropped': 16,
        'Done': 17,
        'Customer Accepted': 18,
        'Working as Expected': 19,
        'Functional Gap': 20,
        'Enhancement Required': 21,
        'Documentation Provided': 22,
        'Enhancement implemented': 23,
        'Training Provided': 24,
        'Won\'t Do': 25,
        'On hold': 26
    }

    # we will only try to label our ticket as fixed out of the above labels on our data
    # which means we need to take into consideration only data points labelled as 'fixed'

    # LF1 - try to mark the jira fixed
    @labeling_function()
    def mark_fixed(self, comments):
        KEYS = r"\bticket (mark|fix|done|ok|verify|verified|close|resolve)"
        return self.POSITIVE if re.search(KEYS, comments) else self.ABSTAIN

    # LF2 - try to check if it is not fixed
    @labeling_function()
    def check_fixed(self, comments):
        KEYS = r"\bticket (pending|incorrect|wrong|waiting|open|update|move)"
        return self.NEGATIVE if re.search(KEYS, comments) else self.ABSTAIN

    # LF3 - try to find false negatives
    @labeling_function()
    def find_false_negatives(self, comments):
        KEYS = r"\b(question|why|not issue|not wrong)"
        return self.POSITIVE if re.search(KEYS, comments) else self.ABSTAIN

    # LF4 - try to find false positives
    @labeling_function()
    def find_false_positives(self, comments):
        KEYS = r"\b(not fix|not done|not correct)"
        return self.NEGATIVE if re.search(KEYS, comments) else self.ABSTAIN

    # on a fairly good amount of train data 2 LFs might just work
    # we can now start making the Ls matrix from the LFs by applying every LF on each of the data point
    def make_Ls_matrix(self, data, LFs):
        noisy_labels = np.empty((len(data), len(LFs)))
        for i, row in data.iterrows():
            for j, lf in enumerate(LFs):
                noisy_labels[i][j] = lf(row)
        return noisy_labels

    def get_label(self, text):
        return self.labels.get(text) if self.labels.get(text) is not None else -1

    def function(self):
        # ensure csv data only has fixed and not fixed
        train = pd.read_csv("training csv file")  # large number of unlabelled tickets
        test = pd.read_csv("testing csv file")  # small number of labelled tickets
        LF_set = pd.read_csv("label function csv file")  # labelled tickets for building LF

        LFs = ['mark_fixed', 'check_fixed', 'find_false_negatives', 'find_false_positives']
        LF_names = {1: 'mark_fixed', 2: 'check_fixed', 3: 'find_false_negatives', 4: 'find_false_positives'}

        # We build a matrix of LF votes for each comment ticket
        LF_matrix = self.make_Ls_matrix(LF_set, LFs)

        # Get true labels for LF set
        Y_LF_set = np.array(LF_set['resolution'])

        display(lf_summary(sparse.csr_matrix(LF_matrix),
                           Y=Y_LF_set,
                           lf_names=LF_names.values()))

        print("label coverage: " + label_coverage(LF_matrix))

        mv = MajorityLabelVoter()
        Y_train_majority_votes = mv.predict(LF_matrix)
        print("classification report:\n" + classification_report(Y_LF_set, Y_train_majority_votes))

        Ls_train = self.make_Ls_matrix(train, LFs)

        # You can tune the learning rate and class balance.
        model = LabelModel(k=2, seed=123)
        model.train_model(Ls_train, n_epochs=2000, print_every=1000,
                                lr=0.0001,
                                class_balance=np.array([0.2, 0.8]))
